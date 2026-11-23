from typing import Callable, Union, cast

import gpytorch
import torch

from .gpytorch_model import ABCGPyTorchModel

# InducingPointKernel builds a sparse COO tensor internally (to interpolate
# between the full and inducing-point covariances); torch only checks sparse
# tensor invariants when explicitly told to, and warns on every sparse tensor
# construction until it's told either way. Invariant checking is off by
# default already (this data all comes from our own kernel/inducing-point
# math, not external input, so there's nothing to validate) - disabling it
# explicitly just silences that warning instead of changing behavior.
torch.sparse.check_sparse_tensor_invariants.disable()


class GPyTorchSGP(gpytorch.models.ExactGP, ABCGPyTorchModel):
	supports_batch_fitting: bool = True

	def __init__(self, X: torch.Tensor, y: torch.Tensor, kernel_factory: Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.kernels.Kernel], likelihood: Union[gpytorch.likelihoods.GaussianLikelihood, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.likelihoods.GaussianLikelihood]], induce_size: int):
		# ExactGP.__init__ (and so torch.nn.Module.__init__) must run before
		# ABCGPyTorchModel.__init__'s register_buffer calls, so batch_shape/y
		# here can't depend on anything ABCGPyTorchModel.__init__ computes -
		# `likelihood` (a pre-built Likelihood or a factory, same dual-mode
		# contract as kernel_factory) must therefore be resolved locally too,
		# via the same resolution ABCGPyTorchModel.__init__ uses internally, so
		# ExactGP.__init__ and ABCGPyTorchModel.__init__ end up sharing the exact
		# same likelihood instance rather than a factory being invoked twice.
		batch_shape = ABCGPyTorchModel._batch_shape(y)
		y_squeezed = y.squeeze(-1) if y.shape[-1] == 1 else y
		resolved_likelihood = ABCGPyTorchModel._resolve_likelihood(likelihood, batch_shape, X.shape[-1], X, y_squeezed)
		gpytorch.models.ExactGP.__init__(self, X, y_squeezed, resolved_likelihood)
		ABCGPyTorchModel.__init__(self, X, y, kernel_factory, resolved_likelihood, induce_size)
		self.mean_module = gpytorch.means.ZeroMean()
		self.covar_module = gpytorch.kernels.InducingPointKernel(self.kernel, inducing_points=self.z, likelihood=resolved_likelihood)

	def forward(self, x: torch.Tensor):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # pyright: ignore[reportArgumentType]

	def objective(self):
		objective_function = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self) # pyright: ignore[reportUnknownVariableType]
		output = self(self.X)
		# gpytorch's MarginalLogLikelihood.__call__ stub returns Tensor |
		# Distribution | LinearOperator generically; an ExactMarginalLogLikelihood
		# call always returns a plain scalar-per-batch Tensor at runtime.
		return -cast(torch.Tensor, objective_function(output, self.Y)) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

	def next_online_model(self, X_new: torch.Tensor, y_new: torch.Tensor) -> None:
		raise NotImplementedError("GPyTorchSGP does not support online updates.")
