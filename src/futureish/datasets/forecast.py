from .config import DataConfig
from .._internal.constants import CONTIGUOUS_GROUP, TRAJ_ID
from .base import DatasetBase
from ..utils.polars import AnyFrame
from .predict import PredictDatasetBase
from ..features.input_base import InputFeatureBase
from ..features.target import TargetFeature
from .._internal.constants import TIME_CL


import polars as pl


from datetime import datetime
from typing import Optional, Sequence


class ForecastingDataset(PredictDatasetBase):
	"""
		TODO: implement specific details
        A dataset class for forecasting tasks. It can create prediction trajectories used for forecasting new data with unknown future/ground truth values
	"""
	def create_prediction_trajectory(self, t0: Optional[datetime] = None, t0_offset: Optional[str] = None, pred_len: str = "1d") -> tuple[pl.DataFrame, pl.DataFrame]:
		raise NotImplementedError("This method still_needs to be implemented")
		if t0_offset is not None and t0 is not None:
			raise ValueError("Only one of t0 or t0_offset should be provided, not both.")
		if t0_offset is not None:
			t0 = self.valid_t0.select(pl.col("t0").min().dt.offset_by(t0_offset)).item()
		if t0 is None:
			t0 = self.valid_t0.select(pl.col("t0").min()).item()

		assert t0 in self.valid_t0.to_series(), "t0 should be in valid_t0"
		traj = (
			self.data
			.filter(
					pl.col("time") >= pl.lit(t0).dt.offset_by("-" + self.config.longest_input_length),
					pl.col("time") <= pl.lit(t0).dt.offset_by(pred_len)
				)
			.with_columns(
				pl.when(pl.col("time") == t0).then(pl.lit(t0)).alias("t0"),
				*[
					pl.when(pl.col("time") <= t0).then(f.base_datapoint.column)
					for f in self.config.target_features
				],
			)
			.collect()
		)
		if traj.select(pl.col(CONTIGUOUS_GROUP).n_unique()).item() != 1:
			raise ValueError("Trajectory is not contiguous. Please check the input data and ensure that there are no gaps in the time series.")

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
					f.base_datapoint.column
					for f in self.config.target_features
				]
			)
		)

		return traj, ground_truth

	def create_all_trajectories(self, pred_len: str = "1d") -> pl.DataFrame:
		raise NotImplementedError("This method still needs to be implemented")
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
			.with_columns(
				pl.lit(valid_pred_start.to_series()).implode()
			)
			.with_columns(
				pl.int_ranges(t0_cnt).alias(TRAJ_ID)
			)
			.explode([TRAJ_ID, "t0"], empty_as_null=True)
			.filter(
				pl.col("time") >= pl.col("t0").dt.offset_by("-" + self.config.longest_input_length),
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