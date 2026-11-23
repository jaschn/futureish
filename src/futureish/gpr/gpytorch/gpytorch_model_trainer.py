import itertools
import time
from collections import defaultdict
from dataclasses import dataclass

import gpytorch # pyright: ignore[reportMissingTypeStubs]
import torch
import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Iterator, Optional, Protocol, Union, cast

from tqdm.auto import tqdm as tqdm_auto

from ...modeling.base import ABCModel
from ...modeling.config import FitConfig
from .backends.gpytorch_sgp import GPyTorchSGP
from .backends.gpytorch_model import ABCGPyTorchModel, default_likelihood_factory


class _TorchGPModel(Protocol):
	"""What this wrapper needs from a basemodel_class instance beyond the
	ABCGPyTorchModel contract - every concrete backend satisfies this, either
	by being a real torch.nn.Module (GPyTorchSGP/gpytorch_vgp, via gpytorch's
	ExactGP/ApproximateGP) or by delegating to one through __getattr__
	(gpytorch_online_pep, which forwards to its underlying_model). Declared
	here rather than on ABCGPyTorchModel itself, since making it a required
	base-class method would force gpytorch_online_pep to shadow it and break
	that delegation.
	"""
	likelihood: Any
	def parameters(self) -> Iterator[torch.nn.Parameter]: ...
	def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]: ...
	def train(self, mode: bool = True) -> Any: ...
	def eval(self) -> Any: ...
	def cuda(self, device: Any = None) -> Any: ...

@dataclass(frozen=True)
class GPyTorchFitConfig(FitConfig):
	"""fit()/fit_step() parameters for GPyTorchTrainer.

	iters/time_limit pick the stopping rule: a fixed iteration count, or a
	wall-clock budget in seconds (for online/incremental fits with a per-step
	time budget) - set one or the other, not both meaningfully at once.
	lr/patience/cooldown feed the Adam optimizer and its ReduceLROnPlateau
	scheduler. tqdm/record_history toggle progress display and per-iteration
	parameter history (both add overhead, so default off). callback(model,
	callback_context, iteration), if given, runs once per iteration;
	callback_context is opaque user data forwarded through unchanged.
	"""
	iters: int = 1000
	time_limit: Optional[float] = None
	lr: float = 0.01
	tqdm: bool = False
	record_history: bool = False
	patience: int = 10
	cooldown: int = 10
	callback: Optional[Callable[..., None]] = None
	callback_context: Any = None

class GPyTorchTrainer(ABCModel):
	def __init__(self, induce_size: int, kernel_factory: Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.kernels.Kernel], likelihood: Union[gpytorch.likelihoods.GaussianLikelihood, Callable[[torch.Size, int, torch.Tensor, torch.Tensor], gpytorch.likelihoods.GaussianLikelihood]] = default_likelihood_factory, basemodel_class: type[ABCGPyTorchModel] = GPyTorchSGP, use_cuda: bool = False) -> None:
		ABCModel.__init__(self)
		self.induce_size = induce_size
		self.kernel_factory = kernel_factory
		self.likelihood = likelihood
		self.basemodel_class = basemodel_class
		self.use_cuda = use_cuda
		self.loss_hist: dict[str, list[float]] = defaultdict(list)  # one entry per parameter name, each holding the whole batched (per-feature) loss at that iteration
		# one entry per parameter name, each holding the whole batched
		# (per-feature) tensor at that iteration.
		self.param_hist: dict[str, list[torch.Tensor]] = defaultdict(list)
		self.noise_hist: list[torch.Tensor] = []
		self._basemodel = None
		if not bool(getattr(self.basemodel_class, "supports_batch_fitting", False)):
			raise TypeError(f"{self.basemodel_class.__name__} does not support batch fitting (supports_batch_fitting=False), so it can't be used through GPyTorchTrainer.")

	@property
	def _repr_fields(self) -> dict[str, Any]:
		new_fields: dict[str, Any] = {
			"induce_size": self.induce_size,
			"kernel_factory": self.kernel_factory,
			"likelihood": self.likelihood,
			"basemodel": self.basemodel_class.__name__ if self._basemodel is None else self._basemodel,
			"use_cuda": self.use_cuda,
		}
		return super()._repr_fields | new_fields

	@property
	def batch_shape(self) -> torch.Size:
		return torch.Size([len(self.feature_names)])

	def _stack_y(self, features: dict[str, NDArray[np.float64]]) -> torch.Tensor:
		columns = [np.asarray(features[name], dtype=np.float64).reshape(-1) for name in self.feature_names]
		y = np.stack(columns, axis=0)  # (F, N)
		return torch.tensor(y, dtype=torch.float64).unsqueeze(-1)  # (F, N, 1)

	def _fit(
		self,
		X: NDArray[np.float64],
		features: dict[str, NDArray[np.float64]],
		config: Optional[FitConfig] = None,
	) -> None:
		if set(features.keys()) != set(self.feature_names):
			raise ValueError("a GPyTorchTrainer must be fit with the same set of features every call, since the batch dimension is fixed at construction")

		config = config or GPyTorchFitConfig()
		assert isinstance(config, GPyTorchFitConfig), f"config must be a GPyTorchFitConfig, got {type(config)}"
		iters = config.iters
		time_limit = config.time_limit
		starting_lr = config.lr
		enable_tqdm = config.tqdm
		record_history = config.record_history
		patience = config.patience
		cooldown = config.cooldown
		callback = config.callback
		callback_context = config.callback_context

		dt = torch.float64
		X_tensor = torch.tensor(X, dtype=dt)
		y_tensor = self._stack_y(features)

		if self._basemodel is None:
			self._basemodel = self.basemodel_class(X_tensor, y_tensor, self.kernel_factory, self.likelihood, self.induce_size)
		else:
			self._basemodel.next_online_model(X_tensor, y_tensor)

		# See _TorchGPModel's docstring: basemodel_class instances carry
		# torch.nn.Module-like behavior that ABCGPyTorchModel itself doesn't
		# declare.
		basemodel = cast(_TorchGPModel, self._basemodel)

		last_lr = starting_lr
		optimizer = torch.optim.Adam(basemodel.parameters(), lr=starting_lr)
		# defaults sized for time_limit-based online fits (typically tens of iterations,
		# not the ~1000 the old patience=100/cooldown=500 assumed); override via kwargs
		# for long fixed-iters runs where a larger patience makes sense.
		lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001, factor=0.1, patience=patience, cooldown=cooldown)

		if self.use_cuda:
			self._basemodel = cast(ABCGPyTorchModel, basemodel.cuda())
			basemodel = cast(_TorchGPModel, self._basemodel)

		basemodel.train()
		iterator = itertools.count() if time_limit is not None else range(iters)

		desc = f"submodels (F={len(self.feature_names)})"
		tq = None
		if enable_tqdm and time_limit is not None:
			tq = tqdm_auto(total=time_limit, desc=desc, leave=False, unit="s", bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}{postfix}]") # pyright: ignore[reportAttributeAccessIssue]
		elif enable_tqdm:
			tq = tqdm_auto(total=iters, desc=desc, leave=False)

		start_time = time.monotonic()
		# debug(False) skips gpytorch's internal shape/dtype assertions and
		# lazily_evaluate_kernels(False) skips LazyTensor wrapping - both are pure
		# overhead here since every kernel call is immediately densified anyway
		# (measured ~7% faster per iteration, no effect on the computed values).
		with gpytorch.settings.debug(False), gpytorch.settings.lazily_evaluate_kernels(False):
			for i in iterator:
				elapsed = time.monotonic() - start_time
				if time_limit is not None and elapsed >= time_limit:
					break

				optimizer.zero_grad()
				per_feature_loss = self._basemodel.objective() # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]
				loss = per_feature_loss.sum()
				loss.backward(retain_graph=True) # pyright: ignore[reportUnknownMemberType]
				optimizer.step() # pyright: ignore[reportUnknownMemberType]
				# .item()/.numpy() force a device sync; call them once per iteration
				# and reuse the values instead of re-syncing for the scheduler, the
				# tqdm postfix, and loss_hist separately (measurable on CUDA, where
				# each sync stalls the queued kernels).
				loss_value = loss.item() # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
				per_feature_loss_values = per_feature_loss.detach().cpu().numpy()
				lr_scheduler.step(loss_value)
				if tq is not None:
					postfix = {"loss": loss_value, "lr": lr_scheduler.get_last_lr()[0]}
					if time_limit is not None:
						tq.n = min(time.monotonic() - start_time, time_limit)
						postfix["iters"] = i + 1
						tq.refresh()
					else:
						tq.update(1)
					tq.set_postfix(postfix) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

				for name, value in zip(self.feature_names, per_feature_loss_values):
					self.loss_hist[name].append(float(value))


				if record_history:
					for name, param in basemodel.named_parameters():
						self.param_hist[name].append(param.detach().cpu().clone())

					self.noise_hist.append(basemodel.likelihood.noise.detach().cpu().flatten().clone()) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

				if last_lr != lr_scheduler.get_last_lr()[0]:
					last_lr = lr_scheduler.get_last_lr()[0]
					tqdm_auto.write(f"Iteration {i}: LR = {last_lr}")

				if callback is not None:
					callback(self, callback_context, i)

		if tq is not None:
			if time_limit is not None:
				# Loop ended because the time budget ran out, not because of an
				# error: mark the bar as fully consumed so e.g. tqdm.notebook
				# doesn't mistake n < total for a crash and leave the widget
				# on screen instead of clearing it.
				tq.n = time_limit
				tq.refresh()
			tq.close()

	def _fit_step(
		self,
		X: NDArray[np.float64],
		features: dict[str, NDArray[np.float64]],
		config: Optional[FitConfig] = None,
	) -> None:
		if not self.basemodel_class.supports_online_fitting:
			raise NotImplementedError("The base model does not support online fitting.")
		self.fit(X, features, config=config)

	def _predict(self, X: NDArray[np.float64]) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
		assert self._basemodel is not None, "Model must be fitted before prediction."
		basemodel = cast(_TorchGPModel, self._basemodel)
		X_tensor = torch.tensor(X, dtype=torch.float64)
		if self.use_cuda:
			X_tensor = X_tensor.cuda()

		basemodel.eval()
		basemodel.likelihood.eval()
		with torch.no_grad(), gpytorch.settings.debug(False), gpytorch.settings.lazily_evaluate_kernels(False):
			f_dist = self._basemodel(X_tensor)
			y_dist = basemodel.likelihood(f_dist)

		means = y_dist.mean.cpu().numpy() # (F, N)
		stds = y_dist.stddev.cpu().numpy() # (F, N)

		return {name: (means[i], stds[i]) for i, name in enumerate(self.feature_names)}

	def _predict_mean(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
		return {name: mean for name, (mean, _) in self._predict(X).items()}
