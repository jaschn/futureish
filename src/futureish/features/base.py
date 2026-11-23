import copy
from typing import Any, Optional, Self, Union

from datetime import datetime
import warnings
import polars as pl

from ..datapoints.base import DataPoint
from ..datapoints.time import TimeDataPoint
from .recorded import RecordedDataPoint
from ..scaling.base import ScalerBase
from ..scaling.noop import NoopScaler
from ..utils.polars import AnyFrame, FrameT, eager_item
from .._internal.repr import ReprMixin
from ..scaling.data import ScalerFactory

class FeaturesBase(ReprMixin):
	def __init__(self, base_datapoint: DataPoint, feature_name: str, scaler_factory: Optional[ScalerFactory] = None, debug: bool = False):
		if type(self) is FeaturesBase:
			raise TypeError("FeaturesBase is an abstract class and cannot be instantiated directly.")

		self.base_datapoint = base_datapoint
		self.feature_name = feature_name
		self._scaler_factory: Optional[ScalerFactory] = scaler_factory
		self._scaler: Optional[ScalerBase] = None
		self.debug = debug

		if self.base_datapoint.name == self.feature_name:
			raise ValueError("Feature name must be different from base datapoint name")
		self._time_column_dtype: Optional[pl.Datetime] = None

	def set_time_column_dtype(self, dtype: pl.Datetime):
		if self._time_column_dtype is not None and self._time_column_dtype != dtype:
			raise ValueError("Time column dtype has already been set and cannot be changed")
		if not isinstance(dtype, pl.Datetime):
			raise ValueError(f"Time column dtype must be a polars Datetime type, got {type(dtype)}")
		self._time_column_dtype = dtype

	@property
	def time_column_dtype(self) -> pl.Datetime:
		if self._time_column_dtype is None:
			raise ValueError("Time column dtype has not been set. Please set it before accessing.")
		return self._time_column_dtype

	@property
	def _repr_fields(self) -> dict[str, Any]:
		fields: dict[str, Any] = {
			"feature_name": self.feature_name,
			"base_datapoint": self.base_datapoint,
			"debug": self.debug,
		}
		if self._scaler is None:
			fields["scaler_factory"] = self._scaler_factory.__name__ if self._scaler_factory is not None else None
		else:
			fields["scaler"] = self._scaler
			
		return fields

	def fitted(self, df: AnyFrame) -> Self:
		fitted = copy.deepcopy(self)
		if self._scaler_factory is not None:
			fitted._scaler = self._scaler_factory(df, self.base_datapoint.name, self.feature_name)
		else:
			fitted._scaler = NoopScaler(df, self.base_datapoint.name, self.feature_name)
		return fitted

	@property
	def scaler(self) -> ScalerBase:
		if self._scaler is None:
			raise ValueError("Scaler has not been initialized. Call fitted() first.")
		return self._scaler

	def _scale_feature(self, df: FrameT, over: Optional[Union[str, list[str]]] = None, with_t0: Optional[bool] = None) -> FrameT:
		return self.scaler.scale(df, over=over, with_t0=with_t0)

	def _filter_to_step_size(self, df: FrameT) -> FrameT:
		if isinstance(self.base_datapoint, TimeDataPoint):
			return df

		if not isinstance(self.base_datapoint, RecordedDataPoint):
			raise ValueError(f"Base datapoint must be a RecordedDataPoint or TimeDataPoint, got {type(self.base_datapoint)}")
		
		df = (
			df
			.with_columns(
				(pl.col("time").dt.truncate(self.base_datapoint.step_size.to_string()) == pl.col("time")).alias("is_step_size")
			)
		)
		if self.debug:
			if not eager_item(df, (pl.col("is_step_size").all())):
				warnings.warn(f"{self.base_datapoint.name} with step size {self.base_datapoint.step_size} does not align with the interval of the data")

		reduced_df = df.filter(pl.col("is_step_size")).drop("is_step_size")

		return reduced_df