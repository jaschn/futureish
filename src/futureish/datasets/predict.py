from .._internal.constants import CONTIGUOUS_GROUP, TRAJ_ID
from .base import DatasetBase
from .._internal.constants import TIME_CL
import warnings

import polars as pl


from datetime import datetime
from typing import Optional, Sequence


class PredictDatasetBase(DatasetBase):
	# TODO: strip down class to common interface for all prediction datasets (TestDataset, ForecastingDataset)
	def create_prediction_trajectory(self, t0: Optional[datetime] = None, t0_offset: Optional[str] = None, pred_len: str = "1d") -> tuple[pl.DataFrame, pl.DataFrame]:
		if t0_offset is not None and t0 is not None:
			raise ValueError("Only one of t0 or t0_offset should be provided, not both.")
		if t0_offset is not None:
			t0 = self.valid_t0.select(pl.col("t0").min().dt.offset_by(t0_offset)).item()
		if t0 is None:
			t0 = self.valid_t0.select(pl.col("t0").min()).item()

		if not t0 in self.valid_t0.to_series():
			raise ValueError("t0 should be in valid_t0")
		
		traj = (
			self.data
			.filter(
					pl.col("time") >= pl.lit(t0).dt.offset_by("-" + self.config.longest_input_length.to_string()),
					pl.col("time") <= pl.lit(t0).dt.offset_by(pred_len)
				)
			.with_columns(
				pl.when(pl.col("time") == t0).then("time").alias("t0"),
				*[
					pl.when(pl.col("time") <= t0).then(f)
					for f in self.config.all_target_features_base_names
				],
			)
		)

		if not traj.select(TIME_CL.max()).item() in self.valid_t0.to_series():
			warnings.warn("The end of the trajectory does not align with a valid t0. This may lead to unexpected behavior during prediction.")

		start = traj.select(TIME_CL.min()).item()
		end = traj.select(TIME_CL.max()).item()
		ground_truth = (
			self.data
			.filter(
				pl.col("time").is_between(start, end)
			)
			.select(
				"time",
				*[
					f for f in self.config.all_target_features_base_names
				]
			)
		)

		return traj, ground_truth

	def create_all_trajectories(self, pred_len: str = "1d") -> pl.DataFrame:
		valid_pred_start = self._calc_valid_t_start(pred_len).collect()

		# Join valid_pred_start with self.valid_t0 to ensure that only valid t0 values (since they could be overwritten) are used for trajectory creation
		valid_pred_start = (
			valid_pred_start
			.join(
				self.valid_t0,
				left_on="t0",
				right_on="t0",
				how="inner"
			)
		)

		if valid_pred_start.height == 0:
			raise ValueError("No valid prediction times found. Check your input data and prediction length.")
		
		t0_cnt = valid_pred_start.height
		all_traj = (
			self.data
			.lazy()
			.with_columns(
				pl.lit(valid_pred_start.to_series()).implode()
			)
			.with_columns(
				pl.int_ranges(t0_cnt).alias(TRAJ_ID)
			)
			.explode([TRAJ_ID, "t0"], empty_as_null=True)
			.filter(
				pl.col("time") >= pl.col("t0").dt.offset_by("-" + self.config.longest_input_length.to_string()),
				pl.col("time") <= pl.col("t0").dt.offset_by(pred_len)
			)
			.with_columns(
				*[
					pl.when(pl.col("time") <= pl.col("t0")).then(f.base_datapoint.column)
					for f in self.config.target_features
				],
				pl.when(pl.col("time") == pl.col("t0")).then("t0")
			)
		)
		return all_traj.collect()

	def create_all_single_points(self) -> pl.DataFrame:
		points = (
			self
			.data
			.join(
				self.valid_t0,
				left_on="time",
				right_on="t0",
				how="left",
				coalesce=False,
			)
		)
		return points

	def create_all_single_inputs(self) -> pl.DataFrame:
		return self.config.create_input(self.create_all_single_points())