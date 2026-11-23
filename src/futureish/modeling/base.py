import inspect
from typing import Any, Callable, Optional, TypeVar
import abc
import numpy as np
from numpy.typing import NDArray

from .config import FitConfig
from .._internal.repr import ReprMixin

def _is_abstract_member(member: object) -> bool:
	if isinstance(member, (classmethod, staticmethod)):
		member = member.__func__

	return getattr(member, "__isabstractmethod__", False)

def _is_overridable_member(member: object) -> bool:
	if isinstance(member, (classmethod, staticmethod)):
		member = member.__func__

	return getattr(member, "__overridable__", False)

def _property_overrides_allowed(
	base_property: property,
	subclass_property: property,
) -> tuple[bool, str]:
	for base_accessor, subclass_accessor in zip(
		(base_property.fget, base_property.fset, base_property.fdel),
		(subclass_property.fget, subclass_property.fset, subclass_property.fdel),
	):
		if subclass_accessor is not None and not (
			base_accessor is not None
			and getattr(base_accessor, "__overridable__", False)
		):
			return False, f"Property {base_property} cannot be overridden by {subclass_property}"

	return True, ""

F = TypeVar("F", bound=Callable[..., Any])

def overridable(method: F) -> F:
	setattr(method, "__overridable__", True)
	return method

class ABCModel(ReprMixin, abc.ABC):
	def __init__(self):
		self._feature_names = None
		self.is_fitted = False

	@property
	def feature_names(self) -> list[str]:
		if self._feature_names is None:
			raise ValueError("Model has not been fitted yet. Call fit() first.")
		return self._feature_names

	@property
	@overridable
	def _repr_fields(self) -> dict[str, Any]:
		return {
			"feature_names": self._feature_names,
			"is_fitted": self.is_fitted,
		}

	def __init_subclass__(cls, **kwargs: Any) -> None:
		super().__init_subclass__(**kwargs)

		for name in cls.__dict__:
			if name.startswith("__") and name.endswith("__"):
				continue

			base_member = ABCModel.__dict__.get(name)

			if base_member is None:
				continue

			if _is_overridable_member(base_member):
				continue

			if _is_abstract_member(base_member):
				continue

			if isinstance(base_member, property):
				subclass_member = cls.__dict__[name]
				if not isinstance(subclass_member, property):
					raise TypeError(
						f"{cls.__name__} cannot override concrete ABCModel "
						f"property {name!r} with a non-property member"
					)
				allow_override, reason = _property_overrides_allowed(base_member, subclass_member)
				if not allow_override:
					raise TypeError(
						f"{cls.__name__} cannot override concrete ABCModel "
						f"property {name!r}: {reason}"
					)
				continue

			raise TypeError(
				f"{cls.__name__} cannot override concrete ABCModel member "
				f"{name!r}"
			)
	
	def fit(self, X: NDArray[np.float64], features: dict[str, NDArray[np.float64]], config: Optional[FitConfig] = None) -> None:
		self._feature_names = list(features.keys())
		self._fit(X, features, config=config)
		self.is_fitted = True

	@abc.abstractmethod
	def _fit(self, X: NDArray[np.float64], features: dict[str, NDArray[np.float64]], config: Optional[FitConfig] = None) -> None:
		raise NotImplementedError("_fit method not implemented")

	def fit_step(self, X: NDArray[np.float64], features: dict[str, NDArray[np.float64]], config: Optional[FitConfig] = None) -> None:
		if not self.is_fitted:
			self._feature_names = list(features.keys())

		self._fit_step(X, features, config=config)

		self.is_fitted = True

	@overridable
	def _fit_step(self, X: NDArray[np.float64], features: dict[str, NDArray[np.float64]], config: Optional[FitConfig] = None) -> None:
		raise NotImplementedError("_fit_step method not implemented")

	def predict(self, X: NDArray[np.float64]) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
		assert X.ndim == 2, "Input X must be a 2D array"
		assert not np.isnan(X).any(), "Input X must not contain NaN values"
		return self._predict(X)

	def predict_mean(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
		assert X.ndim == 2, "Input X must be a 2D array"
		assert not np.isnan(X).any(), "Input X must not contain NaN values"
		return self._predict_mean(X)

	def predict_sample(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
		assert X.ndim == 2, "Input X must be a 2D array"
		assert not np.isnan(X).any(), "Input X must not contain NaN values"
		return self._predict_sample(X)

	@abc.abstractmethod
	def _predict_mean(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
		raise NotImplementedError("_predict_mean method not implemented")
	
	@overridable
	def _predict(self, X: NDArray[np.float64]) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
		raise NotImplementedError("_predict method not implemented")

	@overridable
	def _predict_sample(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
		pred_samples = {}
		preds = self._predict(X)
		for i, name in enumerate(self.feature_names):
			mean, std = preds[name]
			pred_samples[name] = mean + std * np.random.randn(*mean.shape)

		return pred_samples