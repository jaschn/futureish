from ..utils.timedeltaoperator import TimeDeltaOperator

from .config import DataConfig
from .dataset import Dataset
from .base import DatasetBase
from .online_training import OnlineTrainingDataset
from .predict import PredictDatasetBase
from .test import TestDataset
from ..features.input_base import InputFeatureBase
from ..features.target import TargetFeature
from ..utils.polars import AnyFrame
from .._internal.constants import TIME_CL


import polars as pl


from datetime import timedelta
from typing import Any, Sequence, Union


class OnlineDataset(Dataset):
	def __init__(
		self,
		raw_data: AnyFrame,
		input_features: Sequence[InputFeatureBase],
		target_features: Sequence[TargetFeature],
		size_of_first_online_window: str,
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
		new_fields: dict[str, Any] = {
			"size_of_first_online_window": self.size_of_first_online_window,
		}
		return super()._repr_fields | new_fields

	def to_train_dataset(self):
		return OnlineTrainingDataset.create_with_config(
			raw_data=self.data,
			config=self.config,
		)

	def split_training_and_test_data(self, training_split: Union[float, str, timedelta, int]):
		train_end_time = self._split_expr_to_time(training_split)

		if (
			train_end_time < self.data.select(self.size_of_first_online_window.offset_column_forward(TIME_CL.min()).to_physical()).item()
		):
			raise ValueError(
				"training_split results in a training window that is larger than the training data. This would mean the datascaler would be fit on data that is not available at the time of prediction."
			)

		return self._split_training_and_test_data(train_end_time, train_cls=OnlineTrainingDataset)