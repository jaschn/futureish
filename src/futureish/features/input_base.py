from ..utils.timedelta import TimeDeltaOperator

from ..datapoints.base import DataPoint

from .base import FeaturesBase
from ..utils.polars import FrameT, eager_row
from ..scaling.data import ScalerFactory

import polars as pl

from typing import Any, Optional, Union

class InputFeatureBase(FeaturesBase):
	def __init__(
		self,
		base_datapoint: DataPoint,
		feature_name: str,
		uses_future_data: bool = False,
		scaler_factory: Optional[ScalerFactory] = None,
		debug: bool = False,
	):
		if type(self) is InputFeatureBase:
			raise TypeError("InputFeatureBase is an abstract class and cannot be instantiated directly. Please use a subclass.")
		super().__init__(base_datapoint, feature_name, scaler_factory, debug=debug)
		self.uses_future_data = uses_future_data

		self._point_names: Optional[list[str]] = None
		self._x_len: Optional[int] = None
		if self.debug:
			sub_features = self.sub_features_names
			if sub_features is not  None:
				self._x_len = len(sub_features)

	@property
	def _repr_fields(self):
		new_fields = {
			"uses_future_data": self.uses_future_data,
		}
		return super()._repr_fields | new_fields
		
	@property
	def sub_features_names(self) -> Optional[list[str]]:
		if self._point_names is not None:
			return self._point_names
		
		self._point_names = self._generate_sub_feature_names()
		return self._point_names

	def _generate_sub_feature_names(self) -> Optional[list[str]]:
		return None

	def check_sparse_input_compatibility(self, dataset_step_size: TimeDeltaOperator) -> None:
		if not self._supports_sparse_input():
			raise ValueError(f"{self.__class__.__name__} does not support sparse input")

	def _supports_sparse_input(self) -> bool:
		return False
	
	def _prepare(self, dataframe: FrameT, grouped: list[str]) -> FrameT:
		dataframe = self._filter_to_step_size(dataframe)
		scaled =  self._scale_feature(dataframe, over=grouped, with_t0=True)
		limited = (
			scaled.select(
				"time",
				*grouped,
				"t0",
				self.feature_name
			)
			.drop_nulls(["time", self.feature_name] + grouped)
		)
		return limited
	
	def create_features(self, dataframe: FrameT, grouped: Optional[Union[str, list[str]]] = None) -> FrameT:
		if grouped is not None:
			if isinstance(grouped, str):
				grouped = [grouped]
		else:
			grouped = []

		dataframe = self._prepare(dataframe, grouped)
		features = self._create_features(dataframe, grouped)

		features = (
			features
			.select(
				"t0",
				*grouped,
				pl.concat_list(pl.exclude("t0", *grouped)).cast(pl.List(pl.Float64)).alias(self.feature_name)
			)
		)

		if self.debug:
			no_nulls = (
				(
					pl.col(self.feature_name).list.drop_nulls().list.len() == pl.col(self.feature_name).list.len().first()
				).all().alias("no_nulls")
			)

			has_data = (
				(pl.col(self.feature_name).len() > 0).alias("has_data")
			)

			x_len_base = (
				pl.col(self.feature_name).list.len().alias("x_len")
			)
			x_len_unique = (
				(x_len_base.unique().len() == 1).alias("x_len_unique")
			)
			x_len = (
				x_len_base.unique().first().alias("x_len")
			)

			checks = eager_row(features, no_nulls, has_data, x_len_unique, x_len)
			if not checks[0]:
				raise ValueError(f"Feature {self.feature_name} has nulls in its list column after aggregation - check input_length and uses_future_data settings")
			if not checks[1]:
				raise ValueError(f"Feature {self.feature_name} has no data after aggregation - check input_length and uses_future_data settings")
			if not checks[2]:
				raise ValueError(f"Feature {self.feature_name} has inconsistent list lengths after aggregation - check input_length and uses_future_data settings")

			if self._x_len is None:
				self._x_len = checks[3]
			else:
				if self._x_len != checks[3]:
					raise ValueError(f"Feature {self.feature_name} has different list lengths after aggregation than previously created features, before: {self._x_len}, now: {checks[3]} - check input_length and uses_future_data settings")
		
		return features

	def _create_features(self, dataframe: FrameT, grouped: list[str]) -> FrameT:
		raise NotImplementedError("Subclasses of InputFeatureBase must implement _create_features() to build the feature from the input dataframe")