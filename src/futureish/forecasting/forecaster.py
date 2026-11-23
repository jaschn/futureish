from typing import Any, Callable, Optional, Literal, Union

import numpy as np
from numpy.typing import NDArray
import polars as pl

from ..datasets.config import DataConfig
from ..datasets.predict import PredictDatasetBase
from ..datasets.train import TrainDataset
from ..utils.polars import AnyFrame, to_eager
from .._internal.repr import ReprMixin
from .inference import InferenceEngine
from ..modeling.base import ABCModel
from ..modeling.config import FitConfig


class Forecaster(ReprMixin):
	def __init__(self, model: ABCModel):
		self.model = model
		self._config: Optional[DataConfig] = None

	@property
	def _repr_fields(self) -> dict[str, Any]:
		return {
			"model": self.model,
			"config": self._config,
		}

	@classmethod
	def from_fitted(cls, model: ABCModel, config: DataConfig) -> "Forecaster":
		Forecaster = cls.__new__(cls)
		Forecaster.model = model
		Forecaster._config = config
		return Forecaster

	@property
	def config(self) -> DataConfig:
		if self._config is None:
			raise ValueError("Forecaster has not been fit yet. Call fit() first.")
		return self._config

	def _extract(self, df: AnyFrame) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
		df = to_eager(df)
		X = df.select(pl.col("X")).to_series().to_numpy()
		features = df.select(pl.exclude("time", "t0", "X")).to_dict()
		return X, {k: v.to_numpy() for k, v in features.items()}
	
	def fit(self, dataset: TrainDataset, config: Optional[FitConfig] = None) -> None:
		if self.model.is_fitted:
			raise ValueError("Model has already been fitted. Use fit_step for incremental fitting if supported by the model.")
		training_data = dataset.create_training_data()
		X, features = self._extract(training_data)
		self.model.fit(X, features, config=config)
		self._config = dataset.config

	def fit_step(self, dataset: TrainDataset, config: Optional[FitConfig] = None) -> None:
		input_df = dataset.create_training_data()
		if self._config is None:
			self._config = dataset.config
		assert self.config == dataset.config, "Forecaster config must match TrainDataset config for fit_step()"
		X, features = self._extract(input_df)
		self.model.fit_step(X, features, config=config)

	def predict(self, dataframe: AnyFrame) -> pl.DataFrame:
		dataframe = to_eager(dataframe)
		assert "X" in dataframe.columns, "Input dataframe must contain 'X' column"
		assert not dataframe.select(pl.col("X").is_null().any()).item(), "'X' column must not contain null values"
		assert self.model is not None, "call fit or fit_step before predicting"

		id_cols = ["t0"]

		dataframe = dataframe.sort(id_cols)
		x = dataframe.select(pl.col("X")).to_series().to_numpy()
		base_df = dataframe.select(id_cols)
		for feature_name, pred in self.model.predict(x).items():
			base_df = base_df.with_columns(
				pl.Series(name=feature_name + "_mean", values=np.array(pred[0])),
				pl.Series(name=feature_name + "_std", values=np.array(pred[1])),
			)

		return base_df

	def _predict_points(self, dataframe: AnyFrame, pred_type: Literal["mean", "sample"], grouped: Optional[Union[str, list[str]]]) -> pl.DataFrame:
		dataframe = to_eager(dataframe)
		assert "X" in dataframe.columns, "Input dataframe must contain 'X' column"
		assert dataframe.get_column("X").null_count() == 0, "'X' column must not contain null values"
		assert self.model is not None, "call fit or fit_step before predicting"

		id_cols = ["t0"]
		if grouped is not None:
			if isinstance(grouped, str):
				grouped = [grouped]
			id_cols.extend(grouped)

		predict_fn = self.model.predict_mean if pred_type == "mean" else self.model.predict_sample

		dataframe = dataframe.sort(id_cols)
		x = dataframe.get_column("X").to_numpy()
		base_df = dataframe[id_cols]
		predictions = predict_fn(x)
		return base_df.with_columns(
			*[pl.Series(name=feature_name, values=np.array(pred)) for feature_name, pred in predictions.items()]
		)

	# TODO: clean up the prediction interface, making sure it makes sense when just a dataframe with t0 is needed or when one with X is needed. Right now it's a bit confusing.

	def predict_mean(self, dataframe: AnyFrame, grouped: Optional[Union[str, list[str]]] = None) -> pl.DataFrame:
		return self._predict_points(dataframe, pred_type="mean", grouped=grouped)

	def predict_sample(self, dataframe: AnyFrame, grouped: Optional[Union[str, list[str]]] = None) -> pl.DataFrame:
		return self._predict_points(dataframe, pred_type="sample", grouped=grouped)

	def predict_autoregressive(self, df: AnyFrame, samples: int = 1, enable_tqdm: bool = False) -> pl.DataFrame:
		if self.model is None or self._config is None:
			raise ValueError("Model has not been fitted yet. Call fit() first.")
		return InferenceEngine(self).predict_autoregressive(df, samples=samples, enable_tqdm=enable_tqdm)

	def predict_single_step(self, df: AnyFrame, with_bounds: bool = False) -> pl.DataFrame:
		if self.model is None or self._config is None:
			raise ValueError("Model has not been fitted yet. Call fit() first.")
		return InferenceEngine(self).predict_single_step(df, with_bounds=with_bounds)

	def calculate_mse(self, dataset: PredictDatasetBase) -> dict[str, float]:
		if self.model is None or self._config is None:
			raise ValueError("Model has not been fitted yet. Call fit() first.")
		return InferenceEngine(self).calculate_mse(dataset)