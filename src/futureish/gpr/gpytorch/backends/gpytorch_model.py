
from typing import Any, Callable, Optional, Union
import gpytorch # type: ignore
import torch
import abc
import numpy as np

from ....modeling.base import ABCModel
from ...._internal.repr import ReprMixin


def default_likelihood_factory(batch_shape: torch.Size, ND: int, X: torch.Tensor, Y: torch.Tensor) -> gpytorch.likelihoods.GaussianLikelihood:
	"""Default `likelihood` factory for the wrapper-level entry points below -
	a plain GaussianLikelihood with 1e-3 initial noise, matching the old
	likelihood_noise=1e-3 default."""
	likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)
	likelihood.noise = torch.tensor(1e-3, dtype=torch.float64)
	return likelihood

def default_kernel_factory(batch_shape: torch.Size, ND: int, X: torch.Tensor, Y: torch.Tensor) -> gpytorch.kernels.Kernel:
	"""Default `kernel` factory for the wrapper-level entry points below - a
	ScaleKernel wrapping a RBFKernel, with 1.0 initial outputscale and 1.0
	initial lengthscale, matching the old kernel_lengthscale=1.0 and
	kernel_outputscale=1.0 defaults."""
	kernel = gpytorch.kernels.ScaleKernel(
		gpytorch.kernels.RBFKernel(ard_num_dims=ND, batch_shape=batch_shape),
		batch_shape=batch_shape,
	)
	return kernel

class ABCGPyTorchModel(ReprMixin, abc.ABC):
	supports_online_fitting: bool = False
	supports_batch_fitting: bool = False

	@property
	def _repr_fields(self) -> dict[str, Any]:
		return {
			"supports_online_fitting": self.supports_online_fitting,
			"supports_batch_fitting": self.supports_batch_fitting,
		}

	@staticmethod
	def _batch_shape(Y: torch.Tensor) -> torch.Size:
		"""batch_shape (one entry per jointly-trained output, empty when there's
		only one) implied by Y's leading dims, once its trailing size-1 dim
		(present when Y hasn't been squeezed yet) is dropped."""
		if Y.shape[-1] == 1:
			Y = Y.squeeze(-1)
		return Y.shape[:-1]

	@staticmethod
	def _make_inducing_points(X: torch.Tensor, induce_size: int, batch_shape: torch.Size) -> torch.Tensor:
		indices = np.random.permutation(X.shape[-2])[0:induce_size]
		z = X[indices]
		if len(batch_shape) > 0:
			# give every jointly-trained output its own independent copy of the
			# same initial points, rather than a single shared (and therefore
			# gradient-tied) set.
			z = z.unsqueeze(0).expand(*batch_shape, *z.shape).clone()
		return z

	@staticmethod
	def _resolve_likelihood(likelihood: Union[gpytorch.likelihoods.GaussianLikelihood, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.likelihoods.GaussianLikelihood]], batch_shape: torch.Size, ND: int, X: torch.Tensor, Y: torch.Tensor) -> gpytorch.likelihoods.GaussianLikelihood:
		"""Same dual-mode contract as `kernel` below: either an already-built
		Likelihood (with whatever noise value/prior/constraint the caller wants
		baked in), or a factory to defer construction until batch_shape is known
		(see GPyTorchModelWrapper). Factored out as a staticmethod, unlike the
		inline kernel resolution below, because GPyTorchSGP needs to resolve its
		likelihood before this __init__ runs (gpytorch.models.ExactGP requires an
		already-built likelihood at construction time) and must call the same
		resolution logic, not risk re-invoking a stateful factory a second time."""
		return likelihood if isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood) else likelihood(batch_shape, ND, X, Y)

	def __init__(self, X: torch.Tensor, Y: torch.Tensor, kernel: Union[gpytorch.kernels.Kernel, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.kernels.Kernel]], likelihood: Union[gpytorch.likelihoods.GaussianLikelihood, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.likelihoods.GaussianLikelihood]], induce_size: Optional[int] = None, z: Optional[torch.Tensor] = None, verify: bool = False):
		if Y.shape[-1] == 1:
			# Y is (*batch, N, 1) - one batch entry per jointly-trained output,
			# see GPyTorchModelWrapper - so drop only the trailing size-1 dim,
			# not every dim, to keep that batch dim intact.
			Y = Y.squeeze(-1)

		if isinstance(self, torch.nn.Module):
			# register as buffers so X/Y move along with the rest of the model on .cuda()/.to()
			self.register_buffer("X", X)
			self.register_buffer("Y", Y)
		else:
			self.X = X
			self.Y = Y

		self.batch_size = Y.shape[:-1]
		self.N = X.shape[-2]
		self.ND = X.shape[-1]

		if z is None:
			assert induce_size is not None, "Either inducing points 'z' or 'induce_size' must be provided."
			z = self._make_inducing_points(X, induce_size, self.batch_size)
		else:
			# the point-count dim is second-to-last, not 0, once z carries a
			# leading batch dimension (one entry per jointly-trained output)
			induce_size = z.shape[-2]

		self.z = z
		self.induce_size = induce_size
		self.M = induce_size
		self.verify = verify
		# `kernel` must be a factory (Callable[[batch_shape, ND], Kernel]) rather
		# than an already-built Kernel instance whenever the batch_shape isn't
		# known ahead of time (see GPyTorchModelWrapper) - but a pre-built
		# Kernel is also accepted directly, e.g. for the PEP compute classes
		# which are exercised directly against known reference kernels/inducing
		# points in tests.
		self.kernel = kernel if isinstance(kernel, gpytorch.kernels.Kernel) else kernel(self.batch_size, self.ND, self.X, self.Y)
		self.likelihood = self._resolve_likelihood(likelihood, self.batch_size, self.ND, self.X, self.Y)

		if self.verify:
			assert self.X.ndim == 2, "Input X must be a 2D tensor (shared, unbatched, across all outputs)."
			assert self.X.shape[0] == self.N, "N must match the number of rows in X."
			assert self.M > 0, "Inducing size must be positive."

	def __call__(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		return self.forward(x)

	@abc.abstractmethod
	def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
		raise NotImplementedError("Subclasses must implement the forward method.")

	@abc.abstractmethod
	def objective(self) -> Union[torch.Tensor, Any]:
		raise NotImplementedError("Subclasses must implement the objective method.")

	def next_online_model(self, X_new: torch.Tensor, y_new: torch.Tensor) -> None:
		raise NotImplementedError("Subclasses must implement the next_online_model method.")

	@classmethod
	def wrapped(cls, induce_size: int, kernel: Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.kernels.Kernel] = default_kernel_factory, likelihood: Union[gpytorch.likelihoods.GaussianLikelihood, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.likelihoods.GaussianLikelihood]] = default_likelihood_factory, use_cuda: bool = False) -> ABCModel:
		from ..gpytorch_model_trainer import GPyTorchTrainer
		return GPyTorchTrainer(induce_size=induce_size, kernel_factory=kernel, likelihood=likelihood, basemodel_class=cls, use_cuda=use_cuda)