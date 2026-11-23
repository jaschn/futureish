from ..utils.timedelta import TimeDeltaOperator

from ..datapoints.recorded import RecordedDataPoint
from .recorded_base import RecordedInputFeatureBase
from .._internal.constants import TIME_CL

import polars as pl


from typing import Optional, cast

from ..scaling.data import ScalerFactory
from ..utils.polars import FrameT


class RecordedInputFeature(RecordedInputFeatureBase):
	def __init__(
		self,
		base_datapoint: RecordedDataPoint,
		feature_name: str,
		input_length: Optional[str] = None,
		uses_future_data: bool = False,
		scaler_factory: Optional[ScalerFactory] = None,
		*,
		debug: bool = False,
	):
		super().__init__(base_datapoint, feature_name, input_length, uses_future_data, scaler_factory=scaler_factory, debug=debug)

	def _generate_sub_feature_names(self) -> list[str]:
		dtype = self._time_column_dtype if self._time_column_dtype is not None else pl.Datetime
		end = TimeDeltaOperator() if not self.uses_future_data else self.base_datapoint.step_size
		names = (
			pl.DataFrame()
			.select(
				self.input_length.offset_column_backward(pl.lit(0).cast(dtype)).alias("start"),
				end.offset_column_forward(pl.lit(0).cast(dtype)).alias("end")
			)
			.select(
				pl.lit(self.feature_name + "[") + (        
					pl.datetime_range(      
							pl.col("start").first(),
							pl.col("end").first(),
							interval=self.base_datapoint.step_size.to_string(),
							closed="both"
						) - pl.lit(0).cast(dtype)
				).dt.to_string()
				+ pl.lit("]")
			)
			.to_series().to_list()
		)
		return names

	def _create_features(self, dataframe: FrameT, grouped: list[str]) -> FrameT:
		if self.input_length.is_zero():
			grouper = dataframe.group_by("time", *grouped)
		else:
			grouper = (
				dataframe
				.rolling(TIME_CL, period=self.input_length.to_string(), closed="both", group_by=grouped)
			)

		features = cast(FrameT, (
			grouper
			.having(pl.col("t0").last().is_not_null())
			.agg(
				pl.col("t0").last(),
				pl.col(self.feature_name)
			)
			.drop("time")
		))

		if self.uses_future_data:
			futures = cast(FrameT, dataframe.select(
				TIME_CL.dt.offset_by("-" + self.base_datapoint.step_size.to_string()).alias("t0"),
				*grouped,
				pl.col(self.feature_name),
			))

			features = features.join(
				futures,
				on=["t0", *grouped],
				how="left",
				suffix="_fut"
			)

		return features