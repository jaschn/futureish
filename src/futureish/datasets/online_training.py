from ..utils.timedelta import TimeDeltaOperator

from .config import DataConfig
from .train import TrainDataset
from ..utils.polars import AnyFrame
from ..features.input_base import InputFeatureBase
from ..features.target import TargetFeature
from .._internal.constants import TIME_CL


import polars as pl


from datetime import timedelta
from typing import Any, Generator, Sequence, Union


class OnlineTrainingDataset(TrainDataset):
	"""
	A dataset class for online training tasks. It is special, because scaling parameters are only fitted on the first window of data, this avoids data leakage from future data points.
	This is used to simulate online training, where the model is trained on a small window of data and then used to make predictions on new data points.
	"""
	def __init__(
		self,
		raw_data: AnyFrame,
		input_features: Sequence[InputFeatureBase],
		target_features: Sequence[TargetFeature],
		size_of_first_online_window: str
	):
		"""
		:param raw_data: The raw data to be used for creating the dataset. This should be a polars.DataFrame with a column named "time" and other columns corresponding to the input and target features.
		:param input_features: List of input features to be used for creating the input data.
		:param target_features: List of target features to be used for creating the target data.
		:param size_of_first_online_window: The size of the first online window. This should be a string that can be parsed by polars to create a TimeDelta.
		"""
		self._initialize(raw_data, input_features, target_features)
		self.config.add_custom_field("size_of_first_online_window", TimeDeltaOperator(size_of_first_online_window))
		first_window = (
			self.data
			.filter(
				TIME_CL <= self.size_of_first_online_window.offset_column_forward(TIME_CL.min())
			)
		)
		self._init_scaling(first_window)

	@property
	def size_of_first_online_window(self) -> TimeDeltaOperator:
		return self.config.get_custom_field("size_of_first_online_window")
	
	@property
	def _repr_fields(self) -> dict[str, Any]:
		new_fields = {
			"size_of_first_online_window": self.size_of_first_online_window
		}
		return super()._repr_fields | new_fields
	
	def iter(self, step_size: Union[str, timedelta, int, TimeDeltaOperator]) -> Generator[TrainDataset, None, None]:
		if isinstance(step_size, int):
			td = self.data.sort(TIME_CL).select(TIME_CL.min() - TIME_CL.get(step_size)).item()
			ts_op = TimeDeltaOperator(td)
		else:
			ts_op = TimeDeltaOperator(step_size)

		if self.size_of_first_online_window > ts_op:
			raise ValueError(
				"step_size must be bigger than size_of_first_online_window"
			)

		return super()._iter(step_size, TrainDataset)