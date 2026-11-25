from ..utils.timedeltaoperator import TimeDeltaOperator

from .config import DataConfig
from .._internal.constants import CONTIGUOUS_GROUP, TIME_CL
from ..features.base import FeaturesBase
from ..utils.polars import AnyFrame, FrameT, add_contiguous_group
from .._internal.repr import ReprMixin
from ..features.input_base import InputFeatureBase
from ..features.target import TargetFeature

import polars as pl

from datetime import datetime, timedelta
from typing import Any, Generator, Optional, Self, Sequence, TypeVar, Union
import warnings


TDatasetbase = TypeVar("TDatasetbase", bound="DatasetBase")

class DatasetBase(ReprMixin):
	"""
	Underlying base class for a dataset. This class is not meant to be used directly, but rather through a more specific subclass.
	"""
	expects_future_data_known: bool = False
	def __init__(
		self,
		raw_data: AnyFrame,
		input_features: Sequence[InputFeatureBase],
		target_features: Sequence[TargetFeature],
		*,
		debug: bool = False,
	):	
		raise NotImplementedError("DatasetBase is an abstract class and cannot be instantiated directly. Use a create with config or use a subclass of DatasetBase instead.")

	def _initialize(self, raw_data: AnyFrame, input_features: Sequence[InputFeatureBase], target_features: Sequence[TargetFeature], debug: bool = False):
		time_column_dtype = raw_data.collect_schema().get("time")
		if time_column_dtype is None:
			raise ValueError("Raw data must contain a 'time' column")
		if not isinstance(time_column_dtype, pl.Datetime):
			raise ValueError("Raw data 'time' column must be of type pl.Datetime")
		config = DataConfig(input_features, target_features, time_column_dtype, debug=debug)
		self._initialize_with_config(raw_data, config)

	def _initialize_with_config(self, raw_data: AnyFrame, config: DataConfig) -> None:
		self._valid_t0: Optional[pl.DataFrame] = None
		self.config = config
		self._data = self._prepare_raw_data(raw_data)
		self._raw_data_len = self._data.height

	@classmethod
	def create_with_config(cls: type[Self], raw_data: AnyFrame, config: DataConfig) -> Self:
		instance = cls.__new__(cls)
		instance._initialize_with_config(raw_data, config)

		return instance

	@classmethod
	def _init_with_t0(cls: type[Self], raw_data: AnyFrame, config: DataConfig, valid_t0: pl.DataFrame) -> Self:
		instance = cls.create_with_config(raw_data=raw_data, config=config)
		assert valid_t0.columns == ["t0"], "valid_t0 DataFrame must have a single column named 't0'"
		instance._valid_t0 = valid_t0
		return instance

	@property
	def _repr_fields(self) -> dict[str, Any]:
		return {
			"config": self.config,
			"raw_data_len": self._raw_data_len,
			"valid_t0_len": self.valid_t0.height,
		}

	@property
	def data(self) -> pl.DataFrame:
		return self._data

	def _create_contiguous_group_features(self, df: FrameT) -> FrameT:
		recorded_datapoints = [f.base_datapoint for f in self.config.input_features] + [f.base_datapoint for f in self.config.target_features]
		df_reduced = add_contiguous_group(df, recorded_datapoints)
		return df_reduced

	def data_with_contiguous_group(self) -> pl.LazyFrame:
		return self._create_contiguous_group_features(self.data.lazy())
	
	@property
	def valid_t0(self) -> pl.DataFrame:
		if self._valid_t0 is None:
			self._valid_t0 = self._calc_valid_t0()
		if self._valid_t0.height <= 0:
			raise ValueError("No valid prediction times found. Check your input data and step size.")
		return self._valid_t0

	def has_valid_t0(self) -> bool:
		if not self._valid_t0 is None:
			return self._valid_t0.height > 0

		_valid_t0 = self._calc_valid_t0()
		if _valid_t0.height > 0:
			self._valid_t0 = _valid_t0
			return True
		else:
			return False

	def data_with_valid_t0(self) -> pl.LazyFrame:
		return self.data.lazy().join(
			self.valid_t0.lazy(),
			left_on="time",
			right_on="t0",
			how="left",
			coalesce=False
		)

	def _prepare_raw_data(self, raw_data: AnyFrame) -> pl.DataFrame:
		if isinstance(raw_data, pl.LazyFrame):
			raw_data = raw_data.collect()

		if "time" not in raw_data.columns:
			raise ValueError("Raw data must contain a 'time' column")

		if CONTIGUOUS_GROUP in raw_data.columns:
			raise ValueError(f"Raw data already contains '{CONTIGUOUS_GROUP}' column, which is reserved for internal use")

		if raw_data.height <= 0:
			raise ValueError("Raw data is empty")

		all_features_base_names = self.config.all_features_base_names

		if not all(f in raw_data.columns for f in all_features_base_names):
			missing_features = [f for f in all_features_base_names if f not in raw_data.columns]
			raise ValueError(f"Raw data is missing required feature columns: {missing_features}")

		raw_data = raw_data.select(
			"time",
			*[pl.col(c).cast(pl.Float64) for c in all_features_base_names if c != "time"]
		)

		raw_data = raw_data.sort("time")

		return raw_data

	def _calc_valid_t0(self) -> pl.DataFrame:
		if self.expects_future_data_known or self.config.input_uses_future_data():
			return self._calc_valid_t_start(self.config.step_size.to_string()).collect()
		else:
			return self._calc_valid_t_start().collect()

	def _calc_valid_t_start(self, pred_len: Optional[str] = None) -> pl.LazyFrame:
		pred_len_op = TimeDeltaOperator(pred_len) 
		times = (
			self
			.data_with_contiguous_group()
			.drop_nulls([f.base_datapoint.name for f in self.config.target_features])
			.select(
				TIME_CL.filter(
					pl.all_horizontal(
						TIME_CL >= self.config.longest_input_length.offset_column_forward(TIME_CL.min()),
						TIME_CL <= pred_len_op.offset_column_backward(TIME_CL.max()),
					).over(CONTIGUOUS_GROUP)
				).alias("t0")
			)
			.sort("t0")
		)

		return times

	def _init_scaling(self, scale_df: AnyFrame) -> None:
		self.config = self.config.fitted(scale_df)

	def split(self: TDatasetbase, split_expr: Union[float, str, timedelta, int]) -> tuple[TDatasetbase, TDatasetbase]:
		split_time = self._split_expr_to_time(split_expr)
		before_data, after_data = self._split(split_time)

		before_dataset = self.__class__.create_with_config(
			raw_data=before_data,
			config=self.config,
		)

		after_dataset = self.__class__.create_with_config(
			raw_data=after_data,
			config=self.config,
		)

		return before_dataset, after_dataset

	def _split(self, split_time: int, before_data_is_train: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
		if self.expects_future_data_known or self.config.input_uses_future_data() or before_data_is_train:
			before_data = self.data.filter(TIME_CL.dt.offset_by("-" + self.config.step_size.to_string()).to_physical() <= pl.lit(split_time))
		else:
			before_data = self.data.filter(TIME_CL.to_physical() <= pl.lit(split_time))
			
		after_data = self.data.filter(TIME_CL.dt.offset_by(self.config.longest_input_length.to_string()).to_physical() > pl.lit(split_time))

		return before_data, after_data

	def _split_expr_to_time(self, split_expr: Union[float, str, timedelta, int, TimeDeltaOperator]) -> int:
		if isinstance(split_expr, float):
			if not 0 < split_expr <= 1:
				raise ValueError("split must be between 0 and 1")
			data_len = self._raw_data_len
			train_end_idx = int(data_len * split_expr)
			train_end_idx = min(train_end_idx, data_len - 1)
			train_end_idx = max(train_end_idx, 0)
			train_end_time = self.data.select(TIME_CL.get(train_end_idx))
		elif isinstance(split_expr, int): # pyright: ignore[reportUnnecessaryIsInstance]
			if split_expr < 0:
				raise ValueError("split must be a non-negative integer")
			if split_expr >= self._raw_data_len:
				split_expr = self._raw_data_len - 1

			train_end_time = self.data.select(
				TIME_CL.get(split_expr)
			)
		else:
			try:
				split_expr = TimeDeltaOperator(split_expr)
			except Exception as e:
				raise ValueError("training_split must be a float, str, or timedelta or TimeDeltaOperator") from e
			train_end_time = self.data.select(
				split_expr.offset_column_forward(TIME_CL.min())
			)

		if train_end_time.height <= 0:
			raise ValueError("No valid training split time found. Check your input data and step size.")
		if train_end_time.height > 1:
			raise ValueError("Multiple training split times found. Check your input data and step size.")

		train_end_time = train_end_time.select(TIME_CL.to_physical()).item()

		return train_end_time

	def iter(self: TDatasetbase, step_size: Union[str, timedelta, int, TimeDeltaOperator]) -> Generator[TDatasetbase, None, None]:
		return self._iter(step_size, self.__class__)

	def _iter(self, step_size: Union[str, timedelta, int, TimeDeltaOperator], cls: type[TDatasetbase]) -> Generator[TDatasetbase, None, None]:
		if isinstance(step_size, float):
			raise ValueError(f"step_size must be a str, timedelta, or int, not a {type(step_size).__name__}")

		while True:
			end_time = self._split_expr_to_time(step_size)
			chunk, remainder = self._split(end_time)

			chunk_dataset = cls.create_with_config(
				raw_data=chunk,
				config=self.config,
			)

			# maybe a chunk without any valid t0
			if chunk_dataset.has_valid_t0():
				yield chunk_dataset

			if remainder.height <= 0:
				break

			self = self.__class__.create_with_config(
				raw_data=remainder,
				config=self.config,
			)

			#no valid t0 left in the remainder, so we can stop iterating
			if not self.has_valid_t0():
				break

	def random_subset(self: TDatasetbase, t0_count: int, pred_len: Optional[str] = None, random_seed: int = 42) -> TDatasetbase:
		if t0_count <= 0:
			raise ValueError("t0_count must be a positive integer")
		
		if pred_len is None:
			all_t0 = self.valid_t0 # pred_len would be set equal to self.config.step_size, so we can just use the cached valid_t0
		else:
			all_t0 = self._calc_valid_t_start(pred_len).collect()

		if all_t0.height <= t0_count:
			warnings.warn(f"t0_count {t0_count} is larger than the number of valid t0 {all_t0.height}")
			t0_count = all_t0.height

		subset_t0 = all_t0.sample(n=t0_count, with_replacement=False, seed=random_seed)

		subset = self.__class__._init_with_t0(
			raw_data=self.data,
			config=self.config,
			valid_t0=subset_t0,
		)

		return subset

	def evaluate_corr(self) -> pl.DataFrame:
		df = self.data_with_valid_t0()

		features = self.config.create_XY(df)

		if self.config.sub_features_names is None:
			raise ValueError("sub_features_names must be set in the config to evaluate correlation")
		
		corr = (
			features
			.select(
				pl.col("X").arr.to_struct(self.config.sub_features_names).struct.unnest(),
				*[
					pl.col(f.feature_name).arr.first()
					for f in self.config.target_features
				]
			)
			.corr()
		)

		corr = (
			corr
			.select(
				pl.Series("feature_name", corr.columns),
				*[
					pl.col(f.feature_name)
					for f in self.config.target_features
				]
			)
			.filter(
				*[
					pl.col("feature_name") != f.feature_name for f in self.config.target_features
				]
			)
		)

		return corr
		