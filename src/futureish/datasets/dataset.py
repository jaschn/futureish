from .config import DataConfig
from .train import TrainDataset
from ..features.input_base import InputFeatureBase
from ..features.target import TargetFeature
from .base import DatasetBase
from .online_training import OnlineTrainingDataset
from .predict import PredictDatasetBase
from .test import TestDataset
from ..utils.polars import AnyFrame
from .._internal.constants import TIME, TIME_CL


import polars as pl


from datetime import datetime, timedelta
from typing import Any, Sequence, Union


class Dataset(DatasetBase):
	"""
	A basic dataset. This class can be split into training and test datasets or converted into a training dataset.
	"""
	def __init__(
		self,
		raw_data: AnyFrame,
		input_features: Sequence[InputFeatureBase],
		target_features: Sequence[TargetFeature],
		*,
		debug: bool = False,
	):
		"""
		:param raw_data: The raw data to be used for creating the dataset. This should be a polars.DataFrame with a column named "time" and other columns corresponding to the input and target features.
		:param input_features: List of input features to be used for creating the input data.
		:param target_features: List of target features to be used for creating the target data.
		:param debug: If True, additional checks will be performed to ensure that the input and target features are compatible with the raw data. This may slow down the processing, but is useful for debugging.
		"""
		self._initialize(raw_data, input_features, target_features, debug)
		
	def to_train_dataset(self):
		if not self.config.is_fitted:
			self._init_scaling(self.data)
			
		return TrainDataset.create_with_config(
			raw_data=self.data,
			config=self.config,
		)

	def _split_training_and_test_data(self, train_end_time: int, train_cls: type[TrainDataset] = TrainDataset, test_cls: type[TestDataset] = TestDataset) -> tuple[TrainDataset, TestDataset]:
		train_data, other_data = self._split(train_end_time, before_data_is_train=True)

		if not self.config.is_fitted:
			self._init_scaling(train_data)

		return (
			train_cls.create_with_config(
				raw_data=train_data,
				config=self.config,
			),
			test_cls.create_with_config(
				raw_data=other_data,
				config=self.config
			)
		)
	
	def split_training_and_test_data(self, training_split: Union[float, str, timedelta, int]):
		train_end_time = self._split_expr_to_time(training_split)

		return self._split_training_and_test_data(train_end_time)