from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Union, Optional

from ..datasets import PredictDatasetBase
from ..modeling.base import ABCModel

if TYPE_CHECKING:
	from .forecaster import Forecaster

import polars as pl

from ..features.recorded import RecordedInputFeature
from ..features.recorded_base import RecordedInputFeatureBase
from .._internal.constants import SAMPLE_ID, TIME, TRAJ_ID, TIME_CL
from ..datasets.config import DataConfig
from ..features.target import TargetFeature
from ..utils.polars import AnyFrame, eager_row, frame_columns, to_eager
from .._internal.repr import ReprMixin

class InferenceEngine(ReprMixin):
	def __init__(self, Forecaster: "Forecaster"):
		self.forecaster = Forecaster

	@property
	def _repr_fields(self) -> dict[str, Any]:
		return {
			"Forecaster": self.forecaster,
		}

	@property
	def config(self) -> DataConfig:
		return self.forecaster.config

	def _update_features(self, df: pl.DataFrame, new_data: pl.DataFrame, grouped: list[str]) -> pl.DataFrame:
		lazy_df = df.lazy()
		join_key_left = ["time"]
		join_key_right = ["t1"]

		join_key_left.extend(grouped)
		join_key_right.extend(grouped)

		new_data_lazy = new_data.lazy().select(pl.col("t0").dt.offset_by(self.config.step_size.to_string()).alias("t1"), pl.exclude("t0", "time"))

		lazy_df = (
			lazy_df
			.join(
				new_data_lazy,
				left_on=join_key_left,
				right_on=join_key_right,
				how="left",
			)
		)
		
		lazy_df = (
			lazy_df
			.select(
				*join_key_left,
				"t0",
				*[
					pl.coalesce(
							target_feature.base_datapoint.name, # previous data
							target_feature.feature_name, # predicted data
						).alias(target_feature.feature_name)
					for target_feature in self.config.target_features
				],
				*set([
					input_feature.base_datapoint.name
					for input_feature in self.config.input_features
					if isinstance(input_feature, RecordedInputFeatureBase)
				])
			)
		)

		tp1 = pl.col("t0").dt.offset_by(self.config.step_size.to_string()).drop_nulls().first()
		t1_expr = pl.when(TIME_CL == tp1).then(TIME_CL).otherwise(None).alias("t0")

		if len(grouped) > 0:
			t1_expr = t1_expr.over(grouped)

		meta_cols: list[Union[str, pl.Expr]] = [
				TIME_CL,
				t1_expr,
			]

		if len(grouped) > 0:
			meta_cols.extend(grouped)

		for target_feature in self.config.target_features:
			lazy_df = target_feature.scaler.inverse_scale(lazy_df, over=grouped)

		for target_feature in self.config.target_features:
			lazy_df = target_feature.past_predict_step(lazy_df)

		target_base_names = {target_feature.base_datapoint.name for target_feature in self.config.target_features}

		cols: set[str] = set(
				[
					*[
						target_feature.base_datapoint.name
						for target_feature in self.config.target_features
					],
					*[
						input_feature.base_datapoint.name
						for input_feature in self.config.input_features
						if input_feature.base_datapoint.name not in target_base_names
					]
				]
			)
		if TIME in cols:
			cols.remove(TIME)

		lazy_df = (
			lazy_df
			.sort(TIME_CL)
			.select(
				*meta_cols,
				*cols
			)
		)

		result = lazy_df.collect()

		if len(grouped) > 0:
			violation_expr = (pl.col("t0").drop_nulls().n_unique().over(grouped) > 1).any()
		else:
			violation_expr = pl.col("t0").drop_nulls().n_unique() > 1

		time_match_expr = (pl.col("t0").is_null() | (pl.col("time") == pl.col("t0"))).all()

		violation, time_matches = eager_row(result, violation_expr.alias("_violation"), time_match_expr.alias("_time_match"))
		if violation:
			raise ValueError("t0 values are not unique after prediction step. Check your input data and step size.")
		if not time_matches:
			raise ValueError("t0 values do not match time values after prediction step. Check your input data and step size.")

		return result

	def _input_window(self, df: pl.DataFrame, grouped: list[str]) -> pl.LazyFrame:
		"""Trim `df` to just the trailing window create_input() actually needs
		for the single active t0 in this recursive step: longest_input_length
		of history plus one step_size ahead (for uses_future_data features).
		"""
		t0_expr = pl.col("t0").drop_nulls().first()
			
		lower_bound = t0_expr.dt.offset_by("-" + self.config.longest_input_length.to_string()).alias("lower_bound")
		upper_bound = t0_expr.dt.offset_by(self.config.step_size.to_string()).alias("upper_bound")

		window_filter_expr = pl.col("time").is_between(lower_bound, upper_bound, closed="both")

		if len(grouped) > 0:
			window_filter_expr = window_filter_expr.over(grouped)

		df_lazy = (
			df
			.lazy()
			.filter(
				window_filter_expr
			)
		)
		return df_lazy

	def _predict_step(self, df: pl.DataFrame, pred_type: Literal["mean", "samples"], grouped: list[str]) -> pl.DataFrame:
		window_df = self._input_window(df, grouped)
		if pred_type == "mean":
			input_features = self.config.create_input(window_df, grouped)
			outputs = self.forecaster.predict_mean(input_features, grouped)
		else:
			input_features = self.config.create_input(window_df, grouped)
			outputs = self.forecaster.predict_sample(input_features, grouped)
		return self._update_features(df, outputs, grouped)

	def _unscale_one_step(self, target_feature: TargetFeature, anchor_df: pl.DataFrame, preds: pl.DataFrame, value_col: str) -> pl.DataFrame:
		scaler = target_feature.scaler
		raw_name = target_feature.base_datapoint.name
		feature_col = target_feature.feature_name
		null_t0 = pl.lit(None).cast(self.config.time_column_dtype).alias("t0")

		anchor_rows = anchor_df.select(
			TIME_CL, "t0", pl.col("t0").alias("_group"),
			pl.col(raw_name).cast(pl.Float64).alias(feature_col),
		)
		pred_rows = preds.select(
			TIME_CL, null_t0, pl.col("t0").alias("_group"),
			pl.col(value_col).alias(feature_col),
		)

		combined = pl.concat([anchor_rows, pred_rows])
		inverted = scaler.inverse_scale(combined, over="_group")
		return inverted.filter(pl.col("t0").is_null()).select(pl.col("_group").alias("t0"), pl.col(raw_name))

	def predict_single_step(self, df: AnyFrame, with_bounds: bool = False) -> pl.DataFrame:
		df = to_eager(df)
		anchor_df = df.filter(pl.col("t0").is_not_null())

		input_features = self.config.create_input(df)
		if with_bounds:
			preds = self.forecaster.predict(input_features)
		else:
			preds = self.forecaster.predict_mean(input_features)
		
		preds = preds.with_columns(
			pl.col("t0").dt.offset_by(self.config.step_size.to_string()).alias(TIME)
		)

		result = preds.select("t0", "time")
		for target_feature in self.config.target_features:
			if with_bounds:
				mean_col = target_feature.feature_name + "_mean"
			else:
				mean_col = target_feature.feature_name

			raw_name = target_feature.base_datapoint.name

			real_mean = self._unscale_one_step(target_feature, anchor_df, preds, mean_col).rename({raw_name: raw_name + "_mean"})

			result = (
				result
				.join(real_mean, on="t0")
			)

			if with_bounds:
				std_col = target_feature.feature_name + "_std"
				upper_col = target_feature.feature_name + "_upper"

				preds_with_upper = preds.with_columns((pl.col(mean_col) + pl.col(std_col)).alias(upper_col))

				real_upper = self._unscale_one_step(target_feature, anchor_df, preds_with_upper, upper_col).rename({raw_name: raw_name + "_upper"})

				result = (
					result
					.join(real_upper, on="t0")
					.with_columns((pl.col(raw_name + "_upper") - pl.col(raw_name + "_mean")).abs().alias(raw_name + "_std"))
					.drop(raw_name + "_upper")
				)

		return result

	def predict_autoregressive(self, df: AnyFrame, samples: int = 1, enable_tqdm: bool = False) -> pl.DataFrame:
		with pl.Config(set_engine_affinity="in-memory"):
			return self._predict_autoregressive(df, samples=samples, enable_tqdm=enable_tqdm)

	def _predict_autoregressive(self, df: AnyFrame, samples: int, enable_tqdm: bool) -> pl.DataFrame:
		cols = frame_columns(df)
		lazy_df = df.lazy()
		targets = self.config.target_features
		target_base_names = [target_feature.base_datapoint.name for target_feature in targets]
		checked_input_features = [
			input_features.base_datapoint.name
			for input_features in self.config.input_features
			if isinstance(input_features, RecordedInputFeatureBase) and not input_features.needs_future_data_predicted and input_features.base_datapoint.name not in target_base_names
		]

		if not "t0" in cols:
			raise ValueError("Input DataFrame must contain a 't0' column marking the starting time of each prediction step")

		if SAMPLE_ID in cols:
			raise ValueError(f"Input DataFrame must not contain a '{SAMPLE_ID}' column - that is reserved for internal use")

		if not TRAJ_ID in cols:
			lazy_df = lazy_df.with_columns(pl.lit(0).alias(TRAJ_ID))

		only_one_t0 = (
			(pl.col("t0").drop_nulls().len() == 1).over(TRAJ_ID).all().alias("_one_t0")
		)

		longest_traj = (
			pl.datetime_range(
				start=pl.col("t0").drop_nulls().first(),
				end=pl.col("time").max(),
				interval=self.config.step_size.to_string(),
			).len().over(TRAJ_ID).max().alias("_auto_steps")
		)

		if len(checked_input_features) == 0:
			input_data_exists = pl.lit(True).alias("_input_exists")
		else:
			input_data_exists = (
				pl.all_horizontal(
					*[
						pl.col(c).is_not_null().all() for c in checked_input_features
					]
				).alias("_input_exists")
			)

		t0_valid = (
			(TIME_CL == pl.col("t0")).drop_nulls().all().alias("_t0_valid")
		)

		t0_aligns_with_target = (
			pl.all_horizontal(
				pl.col("t0").is_not_null(),
				*[
					pl.col(name).is_not_null() for name in target_base_names
				]
			).any()
		)

		one_t0, auto_steps, input_exists, t0_valid, t0_aligns_with_target = eager_row(lazy_df, only_one_t0, longest_traj, input_data_exists, t0_valid, t0_aligns_with_target)

		if not one_t0:
			raise ValueError(f"Input DataFrame must contain exactly one non-null t0 per trajectory (TRAJ_ID={TRAJ_ID}) for recursive prediction")

		if not input_exists:
			raise ValueError(f"Input DataFrame is missing non-null values for one or more input features: {checked_input_features}")

		if not t0_valid:
			raise ValueError(f"Input DataFrame has invalid t0 values for one or more trajectories: {checked_input_features}")

		if not t0_aligns_with_target:
			raise ValueError(f"Input DataFrame has t0 values that do not align with non-null target values for one or more trajectories: {target_base_names}")

		pred_type = "mean" if samples == 1 else "samples"
		if pred_type == "samples":
			lazy_df = lazy_df.with_columns(pl.int_ranges(0, samples).alias(SAMPLE_ID)).explode(SAMPLE_ID, empty_as_null=True)

		df = lazy_df.collect()

		grouped = [SAMPLE_ID, TRAJ_ID] if pred_type == "samples" else [TRAJ_ID]

		orig_t0 = df.select(TIME_CL, *grouped, pl.col("t0"))

		tq = range(auto_steps)
		if enable_tqdm:
			from tqdm.auto import tqdm
			tq = tqdm(tq, desc="Predicting", unit="step", leave=False)

		for _ in tq:
			df = self._predict_step(df, pred_type=pred_type, grouped=grouped)

		reached_end = (
			df
			.select(
				pl.any_horizontal(
					(pl.col("t0").null_count() == 0),
					(pl.col("t0").drop_nulls().first() == TIME_CL.max()).over(grouped).all()
				)
			).item()
		)
		if not reached_end:
			raise ValueError("Autoregressive prediction did not reach the end of the trajectory. Check your input data and step size.")

		df = (
			df
			.drop("t0")
			.join(
				orig_t0,
				on=[TIME_CL, *grouped],
				how="left",
			)
		)

		if df.select(pl.col(TRAJ_ID).n_unique()).item() == 1:
			df = df.drop(TRAJ_ID)


		return df

	def calculate_mse(self, dataset: PredictDatasetBase) -> dict[str, float]:
		if not dataset.config == self.config:
			raise ValueError("PredictDataset config must match Forecaster config for MSE calculation.")

		single_points = dataset.create_all_single_points()
		preds = self.predict_single_step(single_points, with_bounds=False)
		error = (
			preds
			.join(
				dataset.data,
				on="time",
				how="left",
			)
			.select(
				*[
					(pl.col(c) - pl.col(c + "_mean")).alias(c + "_error") for c in self.config.all_target_features_base_names
				]
			)
		)

		assert error.height > 0, "No matching rows found between predictions and ground truth for MSE calculation."

		mse = (
			error
			.select(
				(pl.all()**2).mean()
			).to_dict()
		)

		mse_dict = {k: v[0] for k, v in mse.items()}

		return mse_dict