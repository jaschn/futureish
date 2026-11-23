from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Optional, Callable, Union
import numpy as np
from numpy.typing import NDArray
import catboost
from ..modeling.base import ABCModel
from ..modeling.config import FitConfig

@dataclass(frozen=True)
class CatboostFitConfig(FitConfig):
	"""
	Configuration for fitting a CatBoost model.
	Besides `enable_tqdm`, all other parameters are passed directly to the CatBoost `fit` method.
	"""
	use_best_model: Optional[bool] = None
	eval_set: Optional[Union[catboost.Pool, list[catboost.Pool], tuple[NDArray, NDArray], list[tuple[NDArray, NDArray]]]] = None
	verbose: Optional[bool] = False
	silent: Optional[bool] = None
	logging_level: Optional[str] = None
	early_stopping_rounds: Optional[int] = None
	enable_tqdm: Optional[bool] = False

class Callback:
	def __init__(self, feature_name: str, iterations: Optional[int] = None):
		self.feature_name = feature_name
		self.iterations = iterations
		from tqdm.auto import tqdm
		self.tqdm = tqdm(total=iterations, desc=f"Training progress {self.feature_name}", leave=False)
	def after_iteration(self, info: SimpleNamespace) -> bool:
		self.tqdm.update(1)
		loss = info.metrics.get("learn", {})
		loss = loss.get("MultiRMSE", loss.get("RMSE", None))
		if loss is not None:
			self.tqdm.set_postfix({"loss": f"{loss[-1]:.6f}"})
		if self.iterations is not None and info.iteration >= self.iterations:
			self.tqdm.close()
		return True  # Continue training

class CatBoostTimeseriesModel(ABCModel):
	"""
	A wrapper around CatBoost for time series forecasting.
	Supports both single-output and multi-output regression.
	"""
	def __init__(self, use_multioutput: bool = False, uncertainty_prediction = False, **catboost_kwargs):
		"""
		:param use_multioutput: If True, uses a single CatBoost model for multi-output regression. If False, uses one model per output.
		:param uncertainty_prediction: If True, enables uncertainty prediction using virtual ensembles. Not compatible with multi-output mode.
		:param catboost_kwargs: Keyword arguments to pass to the CatBoostRegressor
		"""
		super().__init__()
		self._catboost_kwargs = catboost_kwargs
		self.use_multioutput = use_multioutput
		self.uncertainty_prediction = uncertainty_prediction
		self.models: dict[str, catboost.CatBoostRegressor] = {}

		if "allow_writing_files" not in catboost_kwargs:
			catboost_kwargs["allow_writing_files"] = False

		if self.use_multioutput and self.uncertainty_prediction:
			raise ValueError("Uncertainty prediction is not supported with multioutput models.")
		
		if not self.use_multioutput and self._catboost_kwargs.get("loss_function", "RMSE") == "MultiRMSE":
			raise ValueError("When use_multioutput is False, loss_function cannot be 'MultiRMSE'.")
		
		if self.use_multioutput:
			if self._catboost_kwargs.get("loss_function", "MultiRMSE") != "MultiRMSE":
				raise ValueError("When use_multioutput is True, loss_function must be 'MultiRMSE'.")
			self._catboost_kwargs["loss_function"] = "MultiRMSE"
		
		if self.uncertainty_prediction:
			if self._catboost_kwargs.get("loss_function", "RMSEWithUncertainty") != "RMSEWithUncertainty":
				raise ValueError("When uncertainty_prediction is True, loss_function must be 'RMSEWithUncertainty'.")
			self._catboost_kwargs["loss_function"] = "RMSEWithUncertainty"
			
	def _fit(self, X: NDArray[np.float64], features: dict[str, NDArray[np.float64]], config: Optional[FitConfig] = None) -> None:
		enable_tqdm = False
		if config is not None:
			if not isinstance(config, CatboostFitConfig):
				raise TypeError(f"Expected config to be of type CatboostFitConfig, got {type(config)}")
			kwargs = config.as_dict()
			enable_tqdm = kwargs.pop("enable_tqdm", False)
		else:
			kwargs = {}


		if not self.use_multioutput:
			if enable_tqdm:
				from tqdm.auto import tqdm
				iterator = tqdm(self.feature_names, desc="Fitting models")
			else:
				iterator = self.feature_names
			for f in iterator:
				try:
					m = catboost.CatBoostRegressor(**self._catboost_kwargs)
					callbacks = []
					if enable_tqdm:
						callbacks = [Callback(f, iterations=kwargs.get("iterations", None))]
					m.fit(X, features[f], **kwargs, callbacks=callbacks)
					self.models[f] = m
				except Exception as e:
					raise RuntimeError(f"Failed to fit model for feature '{f}': {e}")
		else:
			m = catboost.CatBoostRegressor(**self._catboost_kwargs)
			callbacks = None
			if enable_tqdm:
				callbacks = [Callback("multioutput", iterations=kwargs.get("iterations", None))]
			# features[feature] is a 2d array with shape (n_samples, 1)
			stacked_features = np.concatenate([features[feature] for feature in self.feature_names], axis=1)
			m.fit(X, stacked_features, **kwargs, callbacks=callbacks)
			self.models["multioutput"] = m


	def _predict(self, X: NDArray[np.float64]) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
		if not self.uncertainty_prediction:
			raise ValueError("Uncertainty prediction is not enabled for this model.")
		pred = {}
		for f in self.feature_names:
			pred_all = self.models[f].virtual_ensembles_predict(X, prediction_type="TotalUncertainty", virtual_ensembles_count=10)
			mean = pred_all[:, 0]
			std = np.sqrt(pred_all[:, 1] + pred_all[:, 2])  # Total uncertainty = aleatoric + epistemic
			pred[f] = (mean, std)
		return pred

	def _predict_mean(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
		pred = {}
		if not self.use_multioutput:
			for f in self.feature_names:
				pred_all = self.models[f].predict(X)
				if self.uncertainty_prediction:
					pred[f] = pred_all[:, 0]
				else:
					pred[f] = pred_all
		else:
			stacked_pred = self.models["multioutput"].predict(X)
			if len(self.feature_names) == 1:
				pred[self.feature_names[0]] = stacked_pred.flatten()
			else:
				for i, f in enumerate(self.feature_names):
					pred[f] = stacked_pred[:, i].flatten()
		return pred