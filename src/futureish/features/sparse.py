from ..datapoints.recorded import RecordedDataPoint
from .recorded_base import RecordedInputFeatureBase

from ..utils.polars import FrameT
from ..scaling.data import ScalerFactory
from .._internal.constants import TIME_CL

import polars as pl


from typing import Any, Optional, Union, cast

class SparseInputFeature(RecordedInputFeatureBase):
	def __init__(
		self,
		base_datapoint: RecordedDataPoint,
		feature_name: str,
		input_length: str,
		reduction_factor: int = 2,
		uses_future_data: bool = False,
		scaler_factory: Optional[ScalerFactory] = None,
	):
		super().__init__(base_datapoint, feature_name, input_length, uses_future_data, scaler_factory)
		assert isinstance(reduction_factor, int) and reduction_factor > 0, "reduction_factor must be a positive integer"
		assert not self.input_length.is_zero(), "SparseInputFeature does not support input_length starting with 0"
		self.reduction_factor = reduction_factor

	@property
	def _repr_fields(self) -> dict[str, Any]:
		new_fields: dict[str, Any] = {
			"reduction_factor": self.reduction_factor,
		}
		return super()._repr_fields | new_fields

	def _generate_sub_feature_names(self) -> list[str]:
		names = (
			pl.DataFrame()
			.select(
				self.input_length.offset_column_backward(pl.lit(0).cast(self.time_column_dtype)).alias("start"),
				pl.lit(0).cast(self.time_column_dtype).alias("end")
			)
		)
		data_range = [
			pl.datetime_range(      
				pl.col("start").first(),
				pl.col("end").first(),
				interval=self.base_datapoint.step_size.to_string(),
				closed="both",
			).reverse().gather_every(self.reduction_factor).reverse()
		]
		if self.uses_future_data:
			data_range.append(
				pl.lit(0).cast(self.time_column_dtype).dt.offset_by(self.base_datapoint.step_size.to_string())
			)
		names = (
			names
			.select(
				pl.lit(self.feature_name + "[") + (        
					pl.concat(data_range)
					- pl.lit(0).cast(self.time_column_dtype)
				).dt.to_string()
				+ pl.lit("]")
			)
			.to_series().to_list()
		)

		return names

	def _create_features(self, dataframe: FrameT, grouped: Union[str, list[str]]) -> FrameT:
		features = (
			dataframe
			.rolling(TIME_CL, period=self.input_length.to_string(), closed="both", group_by=grouped)
			.agg(
				pl.col("t0").last(),
				pl.col(self.feature_name).reverse().gather_every(self.reduction_factor).reverse()
			)
			.filter(pl.col("t0").is_not_null())
			.drop("time")
		)

		if self.uses_future_data:
			futures = cast(FrameT, dataframe.select(
				self.base_datapoint.step_size.offset_column_backward(TIME_CL).alias("t0"),
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