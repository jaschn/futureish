from ..modeling.base import ABCModel
from ..modeling.config import FitConfig


import numpy as np
from numpy.typing import NDArray


from typing import Any, Callable, Optional


class DummyModel(ABCModel):
	def __init__(self):
		super().__init__()

	@property
	def _repr_fields(self) -> dict[str, Any]:
		return super()._repr_fields

	def _fit(self, X: NDArray[np.float64], features: dict[str, NDArray[np.float64]], config: Optional[FitConfig] = None) -> None:
		pass

	def _predict(self, X: NDArray[np.float64]) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
		return {name: (np.repeat(np.array([1.0]), X.shape[0]), np.repeat(np.array([0.0]), X.shape[0])) for name in self.feature_names}

	def _predict_mean(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
		return {name: mean for name, (mean, _) in self._predict(X).items()}

	def _predict_sample(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
		return {name: mean for name, (mean, _) in self._predict(X).items()}
