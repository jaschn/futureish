from ..utils.timedeltaoperator import TimeDeltaOperator

from ..datapoints.recorded import RecordedDataPoint
from .input_base import InputFeatureBase

from .base import FeaturesBase
from ..utils.polars import FrameT, eager_row
from ..scaling.data import ScalerFactory

import polars as pl

from typing import Any, Optional, Union

class RecordedInputFeatureBase(InputFeatureBase):
	base_datapoint: RecordedDataPoint

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
		if type(self) is RecordedInputFeatureBase:
			raise TypeError("RecordedInputFeatureBase is an abstract class and cannot be instantiated directly. Please use a subclass.")
		
		self.input_length = TimeDeltaOperator(input_length)
		if self.input_length.td.negative:
			raise ValueError("Input length must be a positive duration")
		
		super().__init__(base_datapoint, feature_name, uses_future_data, scaler_factory, debug=debug)

		if self.uses_future_data and not self.base_datapoint.known_future:
			raise ValueError("Input feature that uses future data must have a base datapoint with known future")

		self.needs_future_data_predicted = not self.base_datapoint.known_future and self.uses_future_data

	@property
	def sub_features_names(self) -> Optional[list[str]]:
		if self._point_names is not None:
			return self._point_names
		
		self._point_names = self._generate_sub_feature_names()
		return self._point_names

	def _generate_sub_feature_names(self) -> Optional[list[str]]:
		return None

	def _supports_sparse_input(self) -> bool:
		return False

	def check_sparse_input_compatibility(self, dataset_step_size: TimeDeltaOperator) -> None:
		if self.base_datapoint.step_size != dataset_step_size:
			target_step_is_multiple_of_datapoint_step_size = dataset_step_size.is_multiple_of(self.base_datapoint.step_size)
			if not target_step_is_multiple_of_datapoint_step_size:
				if not self._supports_sparse_input():
					raise ValueError(f"Input feature {self.feature_name} with step size {self.base_datapoint.step_size} does not support sparse input, but is being used in a dataset with step size {dataset_step_size}")

	@property
	def _repr_fields(self) -> dict[str, Any]:
		new_fields: dict[str, Any] = {
			"input_length": self.input_length,
			"needs_future_data_predicted": self.needs_future_data_predicted,
		}
		return super()._repr_fields | new_fields
