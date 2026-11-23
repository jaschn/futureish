#type: ignore
from datetime import datetime, timedelta

import polars as pl
import pytest

from futureish.datasets.config import DataConfig
from futureish.datasets.online import OnlineDataset
from futureish.features.bucket import BucketInputFeature
from futureish.features.recorded import RecordedInputFeature
from futureish.features.modular import ModularInputFeature
from futureish.datapoints.recorded import RecordedDataPoint
from futureish.features.sparse import SparseInputFeature
from futureish.features.target import TargetFeature
from futureish.scaling.z_score import ZScoreScaler


def make_dataset(row_count: int = 11) -> OnlineDataset:
	data = pl.DataFrame(
		{
			"time": pl.datetime_range(
				datetime(2023, 1, 1),
				datetime(2023, 1, 1) + timedelta(minutes=row_count - 1),
				interval="1m",
				eager=True,
			),
			"value": list(range(row_count)),
		}
	)
	datapoint = RecordedDataPoint("value", step_size="1m", known_future=True)
	return OnlineDataset(
		raw_data=data,
		input_features=[RecordedInputFeature(datapoint, "input", input_length="0m", uses_future_data=False)],
		target_features=[TargetFeature(datapoint, "target")],
		size_of_first_online_window="1m",
	)


def test_feature_constructors_reject_unsupported_modes():
	datapoint = RecordedDataPoint("value", step_size="1m", known_future=True)
	modifier = lambda frame, feature_time, feature, grouped: frame

	with pytest.raises(NotImplementedError, match="ModularInputFeature does not support"):
		ModularInputFeature(datapoint, "input", "1m", True, modifier)

	with pytest.raises(NotImplementedError, match="ModularInputFeature does not support"):
		BucketInputFeature(datapoint, "input", "1m", True, "5m", lambda expr, feature_time: expr)

	with pytest.raises(AssertionError, match="reduction_factor must be"):
		SparseInputFeature(datapoint, "input", "1m", reduction_factor=0)
		
	with pytest.raises(AssertionError, match="does not support input_length"):
		SparseInputFeature(datapoint, "input", "0m")

def test_dataset_split_expression_validation_and_clamping():
	dataset = make_dataset()
	assert dataset._split_expr_to_time("3m") == dataset.data.select(pl.col("time").get(3).to_physical()).item()
	assert dataset._split_expr_to_time(timedelta(minutes=3)) == dataset.data.select(pl.col("time").get(3).to_physical()).item()
	assert dataset._split_expr_to_time(3) == dataset.data.select(pl.col("time").get(3).to_physical()).item()
	assert dataset._split_expr_to_time(100) == dataset.data.select(pl.col("time").get(-1).to_physical()).item()

	with pytest.raises(ValueError, match="between 0 and 1"):
		dataset._split_expr_to_time(1.1)
	with pytest.raises(ValueError, match="non-negative"):
		dataset._split_expr_to_time(-1)
	with pytest.raises(ValueError, match="must be a float"):
		dataset._split_expr_to_time(object())


def test_random_subset_validates_count_and_clamps_large_requests():
	dataset = make_dataset(20)

	with pytest.raises(ValueError, match="positive integer"):
		dataset.random_subset(0)

	with pytest.warns(UserWarning, match="larger than the number of valid t0"):
		subset = dataset.random_subset(100, pred_len="2m")
	assert subset.valid_t0.height == dataset._calc_valid_t_start("2m").collect().height


def test_predict_dataset_validates_trajectory_start_and_prediction_range():
	_, prediction_dataset = make_dataset(20).split_training_and_test_data(0.5)
	first_t0 = prediction_dataset.valid_t0[0, "t0"]

	trajectory, ground_truth = prediction_dataset.create_prediction_trajectory(t0_offset="0m", pred_len="2m")
	assert trajectory[0, "t0"] == first_t0
	assert ground_truth.height == trajectory.height

	with pytest.raises(ValueError, match="Only one of t0 or t0_offset"):
		prediction_dataset.create_prediction_trajectory(t0=first_t0, t0_offset="0m")
	with pytest.raises(ValueError, match="t0 should be in valid_t0"):
		prediction_dataset.create_prediction_trajectory(t0=datetime(2023, 1, 1))
	with pytest.raises(ValueError, match="No valid prediction times found"):
		prediction_dataset.create_all_trajectories(pred_len="100d")


def test_zscore_scaler_handles_constant_columns_without_scaling():
	data = pl.DataFrame({"time": [1, 2], "value": [3.0, 3.0]})
	scaler = ZScoreScaler(data, "value", "scaled")

	assert scaler.std == 1
	assert scaler.scale(data)["scaled"].to_list() == [0.0, 0.0]
