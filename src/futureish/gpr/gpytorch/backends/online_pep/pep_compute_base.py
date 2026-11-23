from ..gpytorch_model import ABCGPyTorchModel

import gpytorch # type: ignore
import linear_operator # type: ignore
import torch


from typing import Callable, Optional, Union, cast


class PEP_Compute_Base(ABCGPyTorchModel, gpytorch.models.GP):
	inducing_points: torch.nn.Parameter
	# register_buffer() itself is untyped w.r.t. the attribute it creates -
	# nn.Module's __getattr__ stub only knows it's Tensor | Module in general
	# - so this class-level declaration is what tells the type checker
	# self.jitter is really a Tensor, same as inducing_points above.
	jitter: torch.Tensor

	def __init__(self, X: torch.Tensor, y: torch.Tensor, kernel: Union[gpytorch.kernels.Kernel, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.kernels.Kernel]], likelihood: Union[gpytorch.likelihoods.GaussianLikelihood, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.likelihoods.GaussianLikelihood]], induce_size: Optional[int] = None, z: Optional[torch.Tensor] = None, verify: bool = False):
		gpytorch.models.GP.__init__(self)  # Initialize the GP base class
		# ABCGPyTorchModel.__init__ resolves `likelihood` (a pre-built Likelihood,
		# e.g. GaussianLikelihood(noise_prior=..., ...) with its noise already
		# set - or a factory deferring construction until batch_shape is known,
		# see GPyTorchModelWrapper) into self.likelihood - batch_shape must match
		# the kernel's (one noise value per jointly-trained output), since a
		# GaussianLikelihood built with no batch_shape at all silently gives every
		# output a single *shared* noise parameter instead of its own.
		ABCGPyTorchModel.__init__(self, X, y, kernel, likelihood, induce_size, z=z, verify=verify)
		self.mean_module = gpytorch.means.ZeroMean()
		self.covar_module = self.kernel

		self.register_parameter("inducing_points", torch.nn.Parameter(self.z))  # Inducing points
		self.alpha = 0.5  # Damping factor for PEP
		self.register_buffer("jitter", torch.tensor([1e-4], dtype=X.dtype))  # Jitter for numerical stability, moves with .to()/.cuda()

		if self.verify:
			assert self.X.ndim == 2, "Input X must be a 2D tensor (shared, unbatched, across all outputs)."
			assert self.Y.shape[-1] == self.X.shape[0], "Input X and output Y must have the same number of rows."
			assert self.inducing_points.ndim >= 2, "Inducing points must be at least a 2D tensor, optionally with a leading batch dimension."
			assert self.X.shape[1] == self.inducing_points.shape[-1], "Input X and inducing points must have the same number of features."
			assert self.alpha > 0 and self.alpha <= 1, "Alpha must be in the range (0, 1]."
			assert self.jitter > 0, "Jitter must be positive for numerical stability."
			assert bool((self.likelihood.noise > 0).all()), "Likelihood noise must be positive."

	@property
	def _repr_fields(self) -> dict[str, object]:
		new_fields: dict[str, object] = {
			"inducing_points": self.inducing_points,
			"mean_module": self.mean_module,
			"covar_module": self.covar_module,
			"likelihood": self.likelihood,
			"alpha": self.alpha,
			"jitter": self.jitter.item(),
		}
		return super(ABCGPyTorchModel, self)._repr_fields | new_fields

	@property
	def sigma2(self) -> torch.Tensor:
		return self.likelihood.noise

	def Kfdiag(self) -> torch.Tensor:
		# diag=True always densifies to a plain Tensor at runtime, though
		# gpytorch's generic __call__ stub types it as Tensor | LinearOperator
		# - same reasoning as Lb()/Lbinv_Kbf() above.
		Kfdiag = cast(torch.Tensor, self.covar_module(self.X, diag=True))
		if self.verify:
			assert Kfdiag.shape[-1] == self.N, "Kfdiag's last dimension must equal the number of data points."
		return Kfdiag

	def Kbf(self):
		Kbf = self.covar_module(self.inducing_points, self.X)
		if self.verify:
			assert Kbf.shape[-2:] == (self.M, self.N), "Kbf's last two dimensions must be (Mb, N)."
		return Kbf

	def Kbb(self):
		Kbb = self.covar_module(self.inducing_points, self.inducing_points) + linear_operator.operators.ConstantDiagLinearOperator(self.jitter, self.M)
		if self.verify:
			assert Kbb.shape[-2:] == (self.M, self.M), "Kbb's last two dimensions must be (Mb, Mb)."
		return Kbb

	def err(self):
		# self.Y is already squeezed to (..., N) - see ABCGPyTorchModel - and
		# mean_module(X) is (..., N) too, so no unsqueeze/squeeze dance is needed.
		err: torch.Tensor = self.Y - self.mean_module(self.X) # type: ignore
		if self.verify:
			assert err.shape[-1] == self.N, "err's last dimension must equal the number of data points."
		return err

	def Lb(self):
		Lb = torch.linalg.cholesky(self.Kbb()) # type: ignore
		Lb = cast(torch.Tensor, Lb)  # Ensure Lb is of type torch.Tensor
		if self.verify:
			assert Lb.shape[-2:] == (self.M, self.M), "Lb's last two dimensions must be (Mb, Mb)."
		return Lb

	def Lbinv_Kbf(self, Lb: torch.Tensor):
		Lbinv_Kbf = torch.linalg.solve_triangular(Lb, self.Kbf().to_dense(), upper=False) # type: ignore
		Lbinv_Kbf = cast(torch.Tensor, Lbinv_Kbf)  # Ensure Lbinv_Kbf is of type torch.Tensor
		if self.verify:
			assert Lbinv_Kbf.shape[-2:] == (self.M, self.N), "Lbinv_Kbf's last two dimensions must be (Mb, N)."
		return Lbinv_Kbf

	def Qff_diag(self, Lbinv_Kbf: torch.Tensor):
		# Mb is the second-to-last dimension (dim -2), not dim 0, once a leading
		# batch dimension is present.
		Qff_diag: torch.Tensor = (Lbinv_Kbf ** 2).sum(dim=-2)
		if self.verify:
			assert Qff_diag.shape[-1] == self.N, "Qff_diag's last dimension must equal the number of data points."
		return Qff_diag

	def Dff(self, Lbinv_Kbf: torch.Tensor):
		Qff_diag = self.Qff_diag(Lbinv_Kbf)
		Dff = self.sigma2 + self.alpha * (self.Kfdiag() - Qff_diag)
		if self.verify:
			assert Dff.shape[-1] == self.N, "Dff's last dimension must equal the number of data points."
		return Dff

	def Lbinv_Kbf_LDff(self, Lbinv_Kbf: torch.Tensor, Dff: torch.Tensor):
		# Dff has no Mb dimension; insert one so it broadcasts against
		# Lbinv_Kbf's (..., Mb, N) shape instead of colliding with a batch dim.
		Lbinv_Kbf_LDff: torch.Tensor = Lbinv_Kbf / Dff.sqrt().unsqueeze(-2)
		if self.verify:
			assert Lbinv_Kbf_LDff.shape[-2:] == (self.M, self.N), "Lbinv_Kbf_LDff's last two dimensions must be (Mb, N)."
		return Lbinv_Kbf_LDff

	def Sinv_y(self, Dff: Union[torch.Tensor, linear_operator.LinearOperator]):
		# self.Y is (..., N) - add back a trailing size-1 dim so this is a
		# column vector for the matmul in c().
		Sinv_y = (self.Y / Dff).unsqueeze(-1) # type: ignore
		Sinv_y = cast(torch.Tensor, Sinv_y)  # Ensure Sinv_y is of type torch.Tensor
		if self.verify:
			assert Sinv_y.shape[-2:] == (self.N, 1), "Sinv_y's last two dimensions must be (N, 1)."
		return Sinv_y

	_bound1_cache = None
	@property
	def _bound1(self):
		if self._bound1_cache is None:
			self._bound1_cache = -0.5 * self.N * torch.log(torch.tensor(2 * torch.pi))
		return self._bound1_cache

	def _maximum_log_likelihood_objective(self) -> torch.Tensor:
		raise NotImplementedError("Subclasses should implement this method.")

	def _add_prior_log_probs(self, mll: torch.Tensor) -> torch.Tensor:
		# PEP's bound is hand-rolled rather than going through gpytorch.mlls, so unlike
		# ExactMarginalLogLikelihood/_ApproximateMarginalLogLikelihood it won't otherwise
		# pick up priors registered on the kernel (e.g. lengthscale_prior=...) or likelihood
		# (noise_prior=...).
		# mll sums a per-point term over all N observations in this batch, so its
		# magnitude (and gradient) scales with N, while a prior contributes one
		# N-independent term. Left unscaled, any fixed-strength prior gets swamped
		# by the likelihood once N is more than a handful of points - multiplying by
		# N keeps a prior's relative influence consistent across differently sized
		# batches (e.g. online fits with varying chunk sizes).
		mll_ndim = mll.ndim
		for _, module, prior, closure, _ in self.named_priors(): # type: ignore
			prior_term = prior.log_prob(closure(module)) * self.N # type: ignore
			mll = mll + prior_term.reshape(*prior_term.shape[:mll_ndim], -1).sum(dim=-1)
		return mll

	def objective(self) -> torch.Tensor:
		mll = self._add_prior_log_probs(self._maximum_log_likelihood_objective())
		return -mll  # Return negative for minimization

	def _predict_f(self, Xnew: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		raise NotImplementedError("Subclasses should implement this method.")

	def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		return self._predict_f(x)