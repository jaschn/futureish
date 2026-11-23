from typing import Sequence

from .config import DataConfig
from .base import DatasetBase
from ..utils.polars import AnyFrame

import polars as pl
from ..features.input_base import InputFeatureBase
from ..features.target import TargetFeature

class TrainDataset(DatasetBase):
	expects_future_data_known: bool = True
	def __init__(
		self,
		raw_data: AnyFrame,
		input_features: Sequence[InputFeatureBase],
		target_features: Sequence[TargetFeature],
	):	
		"""	
		:param raw_data: The raw data to be used for creating the dataset. This should be a polars.DataFrame with a column named "time" and other columns corresponding to the input and target features.
		:param input_features: List of input features to be used for creating the input data.
		:param target_features: List of target features to be used for creating the target data.
		"""
		self._initialize(raw_data, input_features, target_features)
		self._init_scaling(raw_data)
	
	def create_training_data(self) -> pl.DataFrame:
		df = self.data_with_valid_t0()

		features = self.config.create_XY(df)

		return features