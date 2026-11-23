import copy

import gpytorch
import torch
from typing import Any, Callable, Optional, Union, cast

from ..gpytorch_model import ABCGPyTorchModel
from .sgpr_pep import SGPR_PEP
from .osgpr_pep import OSGPR_PEP

class GPyTorchOnlinePEP(ABCGPyTorchModel):
    supports_online_fitting: bool = True
    supports_batch_fitting: bool = True

    def __init__(self, X: torch.Tensor, y: torch.Tensor, kernel_factory: Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.kernels.Kernel], likelihood: Union[gpytorch.likelihoods.GaussianLikelihood, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.likelihoods.GaussianLikelihood]], induce_size: int):
        self.underlying_model = SGPR_PEP(X, y, kernel_factory, likelihood, induce_size)

    @property
    def _repr_fields(self) -> dict[str, Any]:
        new_fields: dict[str, Any] = {
            "underlying_model": self.underlying_model,
        }
        return super(ABCGPyTorchModel, self)._repr_fields | new_fields

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> "GPyTorchOnlinePEP":
        self.underlying_model = self.underlying_model.cuda(device)
        return self

    def _recentered_noise_prior(self) -> Optional[gpytorch.priors.Prior]:
        """Recenter this step's noise_prior at the noise value the fit just
        converged to, so the next step's prior starts from what this step
        learned - the same way mu/Su carry the GP's posterior over inducing
        outputs forward in next_online_model below. Only the prior's
        location/mode moves; its family and spread (as originally configured)
        are left untouched - a GammaPrior stays a GammaPrior, a NormalPrior
        stays a NormalPrior, etc.

        Returns None if the current likelihood has no noise_prior, or a
        GammaPrior's concentration/converged noise leaves it with no interior
        mode to recenter onto (it's then left as-is by the deepcopy in
        next_online_model rather than coerced into a different family).

        Raises NotImplementedError if a noise_prior is set but its family
        isn't one this knows how to recenter - better to surface that than to
        silently leave an unsupported prior un-recentered every step.
        """
        model = self.underlying_model
        noise_covar = model.likelihood.noise_covar
        old_prior = getattr(noise_covar, "noise_prior", None)
        new_noise = noise_covar.noise.detach().clone() # pyright: ignore[reportCallIssue]

        if old_prior is None:
            return None

        if isinstance(old_prior, gpytorch.priors.NormalPrior):
            return gpytorch.priors.NormalPrior(loc=new_noise, scale=old_prior.scale)

        if isinstance(old_prior, gpytorch.priors.GammaPrior):
            # Gamma(concentration, rate) has mode (concentration - 1) / rate for
            # concentration > 1. Keep concentration (the prior's shape/sharpness)
            # fixed and solve for the rate that puts the mode at the converged
            # noise value - the Gamma analogue of moving a Normal's loc.
            concentration = old_prior.concentration
            if torch.any(concentration <= 1) or torch.any(new_noise <= 0):
                return None
            rate = (concentration - 1) / new_noise
            return gpytorch.priors.GammaPrior(concentration=concentration, rate=rate)

        raise NotImplementedError(
            f"_recentered_noise_prior doesn't know how to recenter a {type(old_prior).__name__} noise_prior; "
            "add a case for it above rather than silently leaving it un-recentered."
        )

    def next_online_model(self, X_new: torch.Tensor, y_new: torch.Tensor) -> None:
        Zopt = self.underlying_model.inducing_points.detach().clone()
        pseudo = self.underlying_model(Zopt)
        mu = pseudo.mean.unsqueeze(-1).detach().clone()
        Su = pseudo.covariance_matrix.detach().clone()
        # gpytorch's kernel evaluation sometimes adds a spurious leading
        # singleton dim on top of the model's real batch_shape (one entry per
        # jointly-trained output, empty when there's a single output); strip
        # only that spurious dim, never a genuine batch dim.
        batch_shape = self.underlying_model.covar_module.batch_shape
        expected_ndim = len(batch_shape) + 2
        if Su.ndim == expected_ndim + 1:
            Su = Su.squeeze(0)
        assert Su.ndim == expected_ndim, "Su must have exactly one batch dim per jointly-trained output (plus the two matrix dims)."
        # covar_module(...) is typed Tensor | LinearOperator generically by
        # gpytorch's stubs; a plain kernel call like this one always
        # densifies to a Tensor at runtime (see PEP_Compute_Base for the
        # same pattern with .to_dense()/cast() elsewhere in this backend).
        Kaa = cast(torch.Tensor, self.underlying_model.covar_module(Zopt).detach().clone())
        Zinit = Zopt.detach().clone()
        updated_noise_prior = self._recentered_noise_prior()
        kern = copy.deepcopy(self.underlying_model.covar_module)
        lk = copy.deepcopy(self.underlying_model.likelihood)
        if updated_noise_prior is not None:
            lk.noise_covar.register_prior("noise_prior", updated_noise_prior, "noise")
        alpha = self.underlying_model.alpha
        m_online = OSGPR_PEP(X_new, y_new, kern, mu, Su, Kaa, Zopt, Zinit, alpha, lk)
        self.underlying_model = m_online

    def objective(self) -> Any:
        return self.underlying_model.objective()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return self.underlying_model.forward(x)

    def __getattr__(self, name: str):
        try:
            return getattr(self.underlying_model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
