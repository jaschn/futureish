from datetime import datetime
from typing import Any, Optional

from ..utils.polars import AnyFrame, FrameT, add_contiguous_group, eager_item
from ..utils.polars import eager_row
from ..utils.polars import frame_columns
from .._internal.constants import CONTIGUOUS_GROUP, TIME_CL
from ..scaling.data import ScalerFactory
from .base import FeaturesBase
from ..datapoints.recorded import RecordedDataPoint

import polars as pl


class TargetFeature(FeaturesBase):
	base_datapoint: RecordedDataPoint # reduced type hinting to RecordedDataPoint for clarity

	def __init__(self, base_datapoint: RecordedDataPoint, feature_name: str, scaler_factory: Optional[ScalerFactory] = None, bounds_min: Optional[float] = None, bounds_max: Optional[float] = None, debug: bool = False):
		assert isinstance(base_datapoint, RecordedDataPoint), "base_datapoint must be an instance of RecordedDataPoint"
		super().__init__(base_datapoint, feature_name, scaler_factory=scaler_factory, debug=debug)
		self.bounds_max: Optional[float] = bounds_max
		self.bounds_min: Optional[float] = bounds_min
	
	@property
	def _repr_fields(self) -> dict[str, Any]:
		new_fields: dict[str, Any] = {
			"bounds_min": self.bounds_min,
			"bounds_max": self.bounds_max,
		}
		return super()._repr_fields | new_fields

	def _add_contiguous_group(self, df: FrameT) -> FrameT:
		return add_contiguous_group(df, [self.base_datapoint])

	def create_target(self, dataframe: FrameT) -> FrameT:
		dataframe = dataframe.select("t0", "time", self.base_datapoint.name)

		dataframe = self._filter_to_step_size(dataframe)

		dataframe = self._scale_feature(dataframe, with_t0=False)

		target = (
			dataframe
			.drop_nulls(["time", self.feature_name])
			.sort("time")
			.select(
				pl.col("t0").alias("t0_input"),
				TIME_CL.shift(1).alias("t0"),
				self.feature_name,
			)
			.filter(
				pl.col("t0").is_in(pl.col("t0_input").implode())
			)
			.drop("t0_input")
		)

		if self.debug:
			no_nulls = (
				pl.col(self.feature_name).is_not_null().all()
			)
			if not eager_item(target, no_nulls):
				raise ValueError(f"Target feature {self.feature_name} contains null values. Check your input data and scaler settings.")

		target = (
			target
			.with_columns(
				pl.concat_arr(pl.col(self.feature_name))
			)
		)

		return target

	def past_predict_step(self, df: FrameT) -> FrameT:
		if self.bounds_min is None and self.bounds_max is None:
			return df
		return df.with_columns(
			self.base_datapoint.column.clip(lower_bound=self.bounds_min, upper_bound=self.bounds_max)
		)
