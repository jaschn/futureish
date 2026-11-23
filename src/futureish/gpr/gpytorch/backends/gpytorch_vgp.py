from typing import Callable, Union, cast

from .gpytorch_model import ABCGPyTorchModel

import gpytorch
import torch


class GPyTorchVGP(gpytorch.models.ApproximateGP, ABCGPyTorchModel):
	supports_batch_fitting: bool = True

	def __init__(self, X: torch.Tensor, y: torch.Tensor, kernel_factory: Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.kernels.Kernel], likelihood: Union[gpytorch.likelihoods.GaussianLikelihood, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.likelihoods.GaussianLikelihood]], induce_size: int):
		# ApproximateGP.__init__ (and so torch.nn.Module.__init__) must run
		# before ABCGPyTorchModel.__init__'s register_buffer calls, so
		# batch_shape/z here can't depend on anything ABCGPyTorchModel.__init__
		# computes - z is then handed back to it so both classes agree on the
		# same inducing-point tensor.
		batch_shape = ABCGPyTorchModel._batch_shape(y)
		z = ABCGPyTorchModel._make_inducing_points(X, induce_size, batch_shape)
		variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(induce_size, batch_shape=batch_shape)
		variational_strategy = gpytorch.variational.VariationalStrategy(
			self, z, variational_distribution, learn_inducing_locations=True
		)
		gpytorch.models.ApproximateGP.__init__(self, variational_strategy) # pyright: ignore[reportUnknownMemberType]
		# ABCGPyTorchModel.__init__ resolves `likelihood` (pre-built or a
		# factory, same dual-mode contract as kernel_factory) into self.likelihood.
		ABCGPyTorchModel.__init__(self, X, y, kernel_factory, likelihood, induce_size, z=z)

		self.mean_module = gpytorch.means.ZeroMean()
		self.covar_module = self.kernel

	def forward(self, x: torch.Tensor):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # pyright: ignore[reportArgumentType]

	def objective(self):
		objective_function = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=self.N) # pyright: ignore[reportUnknownVariableType]
		output = self(self.X)
		# See GPyTorchSGP.objective()'s comment: gpytorch's MarginalLogLikelihood
		# call stub is a generic union, but VariationalELBO always returns a
		# plain Tensor here.
		return -cast(torch.Tensor, objective_function(output, self.Y)) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

	def next_online_model(self, X_new: torch.Tensor, y_new: torch.Tensor) -> None:
		raise NotImplementedError("GPyTorchVGP does not support online updates.")
