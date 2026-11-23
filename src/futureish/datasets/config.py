import copy
from typing import Any, Optional, Sequence, TypeVar, Union

from ..datapoints.recorded import RecordedDataPoint
from ..utils.timedelta import TimeDeltaOperator

from ..features.recorded_base import RecordedInputFeatureBase

from .._internal.constants import CONTIGUOUS_GROUP, TIME, TIME_CL
from ..features.base import FeaturesBase
from ..features.target import TargetFeature
from ..utils.polars import AnyFrame, FrameT, eager_row, eager_item, frame_columns
from .._internal.repr import ReprMixin

import polars as pl

from functools import reduce

from ..features.input_base import InputFeatureBase

FeatureT = TypeVar("FeatureT", bound=FeaturesBase)

class DataConfig(ReprMixin):
	"""
	Configuration needed for creating input and target features from raw data. Stores the input and target features and other relevant information about the data.
	This is the underlying information needed for making predictions in combinations with the model.
	"""
	is_fitted: bool = False
	def __init__(
		self,
		input_features: Sequence[InputFeatureBase],
		target_features: Sequence[TargetFeature],
		time_column_dtype: pl.Datetime,
		*,
		debug: bool = False
	):
		"""
		:param input_features: List of input features to be used for creating the input data.
		:param target_features: List of target features to be used for creating the target data.
		:param time_column_dtype: The dtype of the time column in the raw data. This should be a polars.Datetime type, either naive (no timezone) or UTC timezone.
		:param debug: If True, additional checks will be performed to ensure that the input and target features are compatible with the raw data. This may slow down the processing, but is useful for debugging.
		"""
		self._X_input_len_cache: Optional[int] = None
		self.is_fitted = False
		self.debug = debug
		self._additional_fields: dict[str, Any] = {}
		
		self.input_features: Sequence[InputFeatureBase] = input_features
		self.target_features: Sequence[TargetFeature] = target_features
		self.time_column_dtype: pl.Datetime = time_column_dtype
		if not isinstance(self.time_column_dtype, pl.Datetime):
			raise ValueError("time_column_dtype must be a pl.Datetime type")
		if self.time_column_dtype.time_zone not in (None, "UTC"):
			raise ValueError("time_column_dtype must be either naive (no timezone) or UTC timezone")
		
		self._check_setup_inference()
		self.step_size: TimeDeltaOperator = self._get_output_length()
		self.longest_input_length: TimeDeltaOperator = self._get_input_length_extreme()
		self.shortest_input_length: TimeDeltaOperator = self._get_input_length_extreme(shortest=True)
		self._prepare_features()
		self._validate_input_features()
		self._validate_target_features()

	@property
	def _repr_fields(self) -> dict[str, Any]:
		return {
			"input_features": self.input_features,
			"target_features": self.target_features,
			"step_size": self.step_size,
			"time_column_dtype": self.time_column_dtype,
			"longest_input_length": self.longest_input_length,
			"shortest_input_length": self.shortest_input_length,
			"is_fitted": self.is_fitted,
			"debug": self.debug,
			"additional_fields": self._additional_fields,
		}

	@property
	def sub_features_names(self) -> Optional[list[str]]:
		names: list[str] = []
		for feature in self.input_features:
			n = feature.sub_features_names
			if n is not None:
				names.extend(n)
			else:
				return None
		return names

	@property
	def all_input_features_names(self) -> list[str]:
		return [f.feature_name for f in self.input_features]

	@property
	def all_target_features_names(self) -> list[str]:
		return [f.feature_name for f in self.target_features]

	@property
	def all_features_names(self) -> list[str]:
		return self.all_input_features_names + self.all_target_features_names

	@property
	def all_input_features_base_names(self) -> list[str]:
		return list(set([f.base_datapoint.name for f in self.input_features]))

	@property
	def all_target_features_base_names(self) -> list[str]:
		return list(set([f.base_datapoint.name for f in self.target_features]))

	@property
	def all_features_base_names(self) -> list[str]:
		return list(set(self.all_input_features_base_names + self.all_target_features_base_names))

	def input_uses_future_data(self) -> bool:
		return any(f.uses_future_data for f in self.input_features)

	def add_custom_field(self, key: str, value: Any) -> None:
		if key in self._additional_fields:
			raise ValueError(f"Field '{key}' already exists in additional_fields")
		self._additional_fields[key] = value

	def get_custom_field(self, key: str) -> Any:
		if key not in self._additional_fields:
			raise KeyError(f"Field '{key}' does not exist in additional_fields")
		return self._additional_fields[key]

	def remove_custom_field(self, key: str) -> None:
		if key not in self._additional_fields:
			raise KeyError(f"Field '{key}' does not exist in additional_fields")
		del self._additional_fields[key]

	def _validate_input_features(self):
		target_base_names = self.all_target_features_base_names
		for input_feature in self.input_features:
			if not input_feature.base_datapoint.known_future and input_feature.base_datapoint.name not in target_base_names:
				raise ValueError(
					f"Input feature {input_feature.base_datapoint.name} requires future data to be available, the base datapoint does not have future data available and no corresponding target feature is defined."
				)
			
		for f in self.input_features:
			f.check_sparse_input_compatibility(self.step_size)
		
		if self.time_column_dtype.time_unit == "us":
			for f in self.input_features:
				bd = f.base_datapoint
				if isinstance(bd, RecordedDataPoint):
					if bd.step_size.td.ns != 0:
						raise ValueError(
							f"Input feature {f.feature_name} has base datapoint {bd.name} with step size {bd.step_size.to_string()} which is not compatible with time_column_dtype {self.time_column_dtype}. Use a coarser time unit for the step size."
						)

		if self.time_column_dtype.time_unit == "ms":
			for f in self.input_features:
				bd = f.base_datapoint
				if isinstance(bd, RecordedDataPoint):
					if bd.step_size.td.ns != 0 or bd.step_size.td.us != 0:
						raise ValueError(
							f"Input feature {f.feature_name} has base datapoint {bd.name} with step size {bd.step_size.to_string()} which is not compatible with time_column_dtype {self.time_column_dtype}. Use a coarser time unit for the step size."
						)

	def _validate_target_features(self):
		pass

	def _get_all_input_lengths(self) -> list[TimeDeltaOperator]:
		input_lengths: list[TimeDeltaOperator] = []
		for input_feature in self.input_features:
			if isinstance(input_feature, RecordedInputFeatureBase):
				input_lengths.append(input_feature.input_length)

		return input_lengths

	def _get_input_length_extreme(self, shortest: bool = False) -> TimeDeltaOperator:
		input_lengths = self._get_all_input_lengths()
		if len(input_lengths) == 0:
			return TimeDeltaOperator()
		
		return sorted(input_lengths, reverse=shortest)[-1]

	def _get_all_output_lengths(self) -> list[TimeDeltaOperator]:
		output_lengths: list[TimeDeltaOperator] = []
		for target_feature in self.target_features:
			output_lengths.append(target_feature.base_datapoint.step_size)

		return output_lengths

	def _get_output_length(self) -> TimeDeltaOperator:
		if len(self.target_features) == 0:
			return TimeDeltaOperator()

		lens = self._get_all_output_lengths()

		if len(set(lens)) > 1:
			raise ValueError(
				f"All target features must have the same step size, but found {set(lens)}"
			)

		return lens[0]

	def _prepare_features(self):
		self.input_features = sorted(self.input_features, key=lambda f: f.feature_name)
		self.target_features = sorted(self.target_features, key=lambda f: f.feature_name)
		if self.debug:
			for f in self.input_features:
				f.debug = True
			for f in self.target_features:
				f.debug = True
				
		for f in self.input_features:
			f.set_time_column_dtype(self.time_column_dtype)
		for f in self.target_features:
			f.set_time_column_dtype(self.time_column_dtype)

	def _check_setup_inference(self):
		if not self.input_features or len(self.input_features) == 0:
			raise ValueError("No input features defined")
		if not self.target_features or len(self.target_features) == 0:
			raise ValueError("No target features defined")

		all_features: Sequence[FeaturesBase] = list(self.input_features) + list(self.target_features)
		feature_names = [f.feature_name for f in all_features]
		if len(feature_names) != len(set(feature_names)):
			raise ValueError("Feature names must be unique")

	def fitted(self, df: AnyFrame) -> "DataConfig":
		if self.is_fitted:
			raise ValueError("DataConfig is already fitted")
		
		fitted = copy.deepcopy(self)
		columns = frame_columns(df)
		fitted.input_features = [self._fitted_feature(f, df, "input", columns) for f in self.input_features]
		fitted.target_features = [self._fitted_feature(f, df, "target", columns) for f in self.target_features]
		fitted.is_fitted = True
		return fitted

	@staticmethod
	def _fitted_feature(feature: FeatureT, df: AnyFrame, kind: str, columns: list[str]) -> FeatureT:
		if feature.base_datapoint.name not in columns:
			raise ValueError(f"Base datapoint {feature.base_datapoint.name} for {kind} feature {feature.feature_name} not found in raw data")
		return feature.fitted(df)

	def _check_input_length(self, lazy_df: pl.LazyFrame, grouped: list[str]) -> None:
		min_time_expr = TIME_CL.min()

		if len(grouped) > 0:
			min_time_expr = min_time_expr.over(grouped)

		exprs = [
			(min_time_expr > pl.col("t0").dt.offset_by("-" + self.longest_input_length.to_string())).any().alias("_enough_history"),
		]

		duplicate_check_expr = pl.col("time").is_duplicated().any()

		if len(grouped) > 0:
			duplicate_check_expr = duplicate_check_expr.over(grouped).any()

		duplicate_check_expr = duplicate_check_expr.alias("_has_duplicates")

		exprs.append(duplicate_check_expr)

		row = eager_row(lazy_df, *exprs)

		if row[0]:
			raise ValueError("Not enough historical data available before t0 for the configured input length")
		if row[1]:
			raise ValueError("Dataframe contains duplicate time values but no group column specified")

	def _get_X_input_len(self, df: FrameT) -> int:
		if self._X_input_len_cache is not None:
			return self._X_input_len_cache

		# The length of the input vector is the sum of all input features' lengths
		list_len = eager_item(df, pl.col("X").list.len().first())
		assert list_len is not None, "Length of input X can't be determined"
		self._X_input_len_cache = list_len
		return list_len

	def _build_input(
		self, df: FrameT, grouped: Optional[Union[str, list[str]]] = None
	) -> pl.LazyFrame:
		"""Lazy core of create_input() - everything except the final collect.
		Split out so create_XY() can join this with _build_training_target()
		and collect the combined X/Y query once, instead of create_input()
		and create_training_target() each forcing their own separate collect
		of the same underlying data.
		"""
		cols = frame_columns(df)
		if TIME not in cols:
			raise ValueError(f"Dataframe must contain '{TIME}' column for create_input()")
		if not "t0" in cols:
			raise ValueError("Dataframe must contain 't0' column for create_input()")
		schema = df.collect_schema()
		t0_dtype = schema["t0"]
		time_dtype = schema[TIME]
		if t0_dtype != time_dtype or time_dtype != self.time_column_dtype:
			raise ValueError(
				f"Dataframe 't0' column dtype {t0_dtype} and '{TIME}' column dtype {time_dtype} do not match DataConfig time_column_dtype {self.time_column_dtype}"
			)

		if grouped is None:
			grouped = []
		if isinstance(grouped, str):
			grouped = [grouped]

		lazy_df = df.lazy()

		join_key = ["t0"]
		join_key.extend(grouped)

		self._check_input_length(lazy_df, grouped)

		all_input_features = [
			input_feature.create_features(lazy_df, grouped)
			for input_feature in self.input_features
		]

		if self.debug:
			for f in zip(self.input_features, all_input_features):
				dtype = f[1].collect_schema()["t0"]
				if dtype != self.time_column_dtype:
					raise Exception(
						f"Input feature {f[0].feature_name} has time column dtype {dtype}, but expected {self.time_column_dtype}."
					)

		all_input_features_joined = reduce(
			lambda left, right: left.join(right, on=join_key, how="left"),
			all_input_features,
		)

		all_input_features_joined = all_input_features_joined.select(
			*join_key, pl.concat_list(pl.exclude(*join_key)).alias("X")
		)

		list_len = self._get_X_input_len(all_input_features_joined)
		all_input_features_joined_arr = all_input_features_joined.with_columns(
			pl.col("X").list.to_array(list_len)
		).sort("t0")
		
		if self.debug:
			try:
					all_input_features_joined_arr.collect()
			except Exception as e:
				lens = all_input_features_joined.select(pl.col("X").list.len().unique()).collect()
				raise ValueError(
					f"Failed to convert input features to array. Check that all input features have the same length. Error: {e}"
					f" Unique lengths found: {lens.to_dicts()}"
				)

		return all_input_features_joined_arr

	def create_input(
		self, df: FrameT, grouped: Optional[Union[str, list[str]]] = None
	) -> pl.DataFrame:

		input_df = self._build_input(df, grouped).collect()
		return input_df

	def _build_training_target(self, df: FrameT) -> pl.LazyFrame:
		lazy_df = df.lazy()
		all_target_features = [
			target_feature.create_target(lazy_df)
			for target_feature in self.target_features
		]

		reduced_targets = reduce(
			lambda left, right: left.join(right, on="t0", how="left"),
			all_target_features
		)

		return reduced_targets

	def create_training_target(self, df: FrameT) -> pl.DataFrame:
		return self._build_training_target(df).collect()

	def create_XY(self, df: AnyFrame) -> pl.DataFrame:
		lazy_df = df.lazy()
		if "t0" not in frame_columns(lazy_df):
			raise ValueError("Dataframe must contain 't0' column for create_XY()")

		X_lazy = self._build_input(lazy_df)
		Y_lazy = self._build_training_target(lazy_df)
		
		# One collect for both X and Y instead of one each.
		XY = X_lazy.join(Y_lazy, on="t0", how="inner").sort("t0").collect()

		n_rows, all_not_null = eager_row(
			XY,
			pl.len().alias("n_rows") > 0,
			pl.all_horizontal(pl.all().explode().is_not_null().all()).alias("all_not_null"),
		)
		if not n_rows:
			raise ValueError("No data points found for training. Check your input data and step size.")
		if not all_not_null:
			raise ValueError("Null values found in training data. Check your input data and step size.")

		return XY
