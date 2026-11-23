from typing import cast

import polars as pl
from datetime import datetime
from pytest import fixture
import pytest
from polars.testing import assert_frame_equal

from futureish.datasets.online import OnlineDataset
from futureish.datasets.train import TrainDataset
from futureish.features.input_base import InputFeatureBase
from futureish.features.sincos_time import SinCosTimeFeature
from futureish.forecasting.forecaster import Forecaster

@fixture
def basic_data() -> pl.DataFrame:
    dummy_data = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2023, 1, 1, 0, 0, 0),
                end=datetime(2023, 1, 1, 0, 10, 0),
                interval="1m",
                eager=True,
            ),
            "value_a": pl.arange(0, 11, eager=True),
            "value_b": pl.arange(10, 21, eager=True),
            "value_c": pl.arange(20, 31, eager=True),
        }
    )
    return dummy_data

@fixture
def dataset(basic_data: pl.DataFrame) -> TrainDataset:
    from futureish.datasets.online import OnlineDataset
    from futureish.features.recorded import RecordedInputFeature
    from futureish.datapoints.recorded import RecordedDataPoint
    from futureish.features.target import TargetFeature

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")
    dp_c = RecordedDataPoint(name="value_c", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
        RecordedInputFeature(base_datapoint=dp_b, feature_name="B", uses_future_data=False, input_length="1m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C"),
    ]

    # Create dataset and use the training subdataset from a 60% split
    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    train_dataset, _ = online_dataset.split_training_and_test_data(0.6)

    return train_dataset

@fixture
def dataset_scaled(basic_data: pl.DataFrame) -> TrainDataset:
    from futureish.datasets.online import OnlineDataset
    from futureish.features.recorded import RecordedInputFeature
    from futureish.datapoints.recorded import RecordedDataPoint
    from futureish.features.target import TargetFeature
    from futureish.scaling import ZScoreScaler, DiffScaler

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")
    dp_c = RecordedDataPoint(name="value_c", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m", scaler_factory=ZScoreScaler),
        RecordedInputFeature(base_datapoint=dp_b, feature_name="B", uses_future_data=False, input_length="1m", scaler_factory=ZScoreScaler),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C", scaler_factory=DiffScaler),
    ]

    # Create dataset and use the training subdataset from a 60% split
    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    train_dataset, _ = online_dataset.split_training_and_test_data(0.6)

    return train_dataset

@fixture
def basic_data_zscale() -> pl.DataFrame:
    import numpy as np
    dummy_data = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2023, 1, 1, 0, 0, 0),
                end=datetime(2023, 1, 2, 0, 0, 0),
                interval="1m",
                eager=True,
            ),
            "value_a": pl.arange(0, 1441, eager=True),
            "value_b": pl.arange(10, 1451, eager=True),
            "value_c": np.random.normal(loc=1, scale=1, size=1441),
        }
    )
    return dummy_data

@fixture
def dataset_zscale(basic_data_zscale: pl.DataFrame) -> TrainDataset:
    from futureish.datasets.online import OnlineDataset
    from futureish.features.recorded import RecordedInputFeature
    from futureish.datapoints.recorded import RecordedDataPoint
    from futureish.features.target import TargetFeature
    from futureish.scaling import ZScoreScaler

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")
    dp_c = RecordedDataPoint(name="value_c", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="0m", scaler_factory=ZScoreScaler),
        RecordedInputFeature(base_datapoint=dp_b, feature_name="B", uses_future_data=False, input_length="0m", scaler_factory=ZScoreScaler),
        RecordedInputFeature(base_datapoint=dp_c, feature_name="C_feature", uses_future_data=False, input_length="0m", scaler_factory=ZScoreScaler),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C_target", scaler_factory=ZScoreScaler),
    ]

    # Create dataset, sizing the online window to cover the whole span so the
    # scaler is fit over everything, and splitting past the last row (clamped
    # to it) so every valid t0 ends up in training - equivalent to the legacy
    # class's training_split=1.
    time_min: datetime = cast(datetime, basic_data_zscale["time"].min())
    time_max: datetime = cast(datetime, basic_data_zscale["time"].max())
    span_minutes = int((time_max - time_min).total_seconds() // 60)
    online_dataset = OnlineDataset(
        raw_data=basic_data_zscale,
        input_features=input_features,
        target_features=target_features,
        size_of_first_online_window=f"{span_minutes}m",
    )
    train_dataset = online_dataset.to_train_dataset()

    return train_dataset

def test_recursive_forecast_dummy(dataset: TrainDataset, basic_data: pl.DataFrame):
    from futureish._internal.dummy import DummyModel

    forecaster = Forecaster(DummyModel())
    forecaster.fit(dataset)

    t0 = datetime(2023, 1, 1, 0, 3, 0)
    dummy_data_prediction = (
        basic_data
            .select(
                "time",
                pl.col("value_a"),
                pl.col("value_b"),
                pl.when(pl.col("time") < datetime(2023, 1, 1, 0, 3, 0)).then("value_c").otherwise(None),
                pl.when(pl.col("time").dt.offset_by("1m") == t0).then("time").otherwise(None).alias("t0")
            )
    )

    pred = forecaster.predict_autoregressive(dummy_data_prediction)

    cols = [c for c in dummy_data_prediction.columns if c not in ("t0")]

    expected = dummy_data_prediction.with_columns(pl.exclude("time", "t0").fill_null(1)).select(sorted(cols))

    pred = pred.select(sorted(pred.columns)).drop("t0", "traj_id", strict=False)

    assert pred.equals(expected), "Predictions do not match expected values"

def test_recursive_forecast_same_duplicate_input(basic_data: pl.DataFrame):
    from futureish._internal.dummy import DummyModel

    from futureish.datasets.online import OnlineDataset
    from futureish.features.recorded import RecordedInputFeature
    from futureish.datapoints.recorded import RecordedDataPoint
    from futureish.features.target import TargetFeature
    from futureish.scaling import ZScoreScaler, DiffScaler

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_c = RecordedDataPoint(name="value_c", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m", scaler_factory=ZScoreScaler),
        RecordedInputFeature(base_datapoint=dp_a, feature_name="B", uses_future_data=False, input_length="1m", scaler_factory=ZScoreScaler),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C", scaler_factory=DiffScaler),
    ]

    # Create dataset and use the training subdataset from a 60% split
    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    train_dataset, _ = online_dataset.split_training_and_test_data(0.6)

    forecaster = Forecaster(DummyModel())
    forecaster.fit(train_dataset)

    t0 = datetime(2023, 1, 1, 0, 3, 0)
    dummy_data_prediction = (
        basic_data
            .select(
                "time",
                pl.col("value_a"),
                pl.col("value_b"),
                pl.when(pl.col("time") < datetime(2023, 1, 1, 0, 3, 0)).then("value_c").otherwise(None),
                pl.when(pl.col("time").dt.offset_by("1m") == t0).then("time").otherwise(None).alias("t0")
            )
    )

    pred = forecaster.predict_autoregressive(dummy_data_prediction)


def test_predict_raises_on_truncated_input_window(dataset: TrainDataset, basic_data: pl.DataFrame):
    """forecaster.predict() must fail fast with an actionable error when the
    dataframe handed to it doesn't carry enough history before the last known
    target time for an input feature's rolling lookback window to be fully
    populated - instead of silently building a shorter/misaligned feature
    vector (or, with a real model, failing deep inside the kernel with a
    cryptic 'x1_ and x2_ must have the same number of dimensions!' error).
    """
    from futureish._internal.dummy import DummyModel

    forecaster = Forecaster(DummyModel())
    forecaster.fit(dataset)

    t0 = datetime(2023, 1, 1, 0, 6, 0)
    # input_length="1m" on 1-minute data needs the row *before* t0 too (a
    # window of 2 points); trimming the dataframe to start exactly at t0
    # leaves that window one point short.
    truncated_prediction = (
        basic_data
        .filter(pl.col("time") >= t0)
        .select(
            "time",
            pl.col("value_a"),
            pl.col("value_b"),
            pl.when(pl.col("time") <= t0).then("value_c").otherwise(None),
            pl.when(pl.col("time") == t0).then("time").otherwise(None).alias("t0")
        )
    )

    with pytest.raises(ValueError, match="Not enough historical data"):
        forecaster.predict_autoregressive(truncated_prediction)


def test_predict_succeeds_with_sufficient_input_window(dataset: TrainDataset, basic_data: pl.DataFrame):
    """Same truncated scenario as above, but with the one extra point of
    history the input feature's window needs - this must succeed and must
    not be flagged by the window-size check.
    """
    from futureish._internal.dummy import DummyModel

    forecaster = Forecaster(DummyModel())
    forecaster.fit(dataset)

    t0 = datetime(2023, 1, 1, 0, 6, 0)
    buffered_prediction = (
        basic_data
        .filter(pl.col("time") >= datetime(2023, 1, 1, 0, 5, 0))
        .select(
            "time",
            pl.col("value_a"),
            pl.col("value_b"),
            pl.when(pl.col("time") <= t0).then("value_c").otherwise(None),
            pl.when(pl.col("time") == t0).then("time").otherwise(None).alias("t0")
        )
    )

    pred = forecaster.predict_autoregressive(buffered_prediction)

    assert pred.filter(pl.col("time") > t0).select(pl.col("value_c").is_not_null().all()).item()


def test_recursive_forecast_dummy_zscale(dataset_zscale: TrainDataset, basic_data_zscale: pl.DataFrame):
    from futureish._internal.dummy import DummyModel

    forecaster = Forecaster(DummyModel())
    forecaster.fit(dataset_zscale)

    t0 = datetime(2023, 1, 1, 0, 3, 0)

    dummy_data_prediction = (
        basic_data_zscale
            .select(
                "time",
                pl.col("value_a"),
                pl.col("value_b"),
                pl.when(pl.col("time") < t0).then("value_c").otherwise(None),
                pl.when(pl.col("time").dt.offset_by("1m") == t0).then("time").otherwise(None).alias("t0")
            )
    )

    pred = forecaster.predict_autoregressive(dummy_data_prediction)

    pred = pred.filter(pl.col("time") >= t0)

    for i in range(pred.height):
        assert pred[i, "value_c"] == pytest.approx(2.0, abs=0.1), f"Prediction at index {i} does not match expected value"


def test_predict_one_step_unscales_mean_and_std(dataset_scaled: TrainDataset, basic_data: pl.DataFrame):
    """InferenceModel.predict() (unlike predict_autoregressive(), which
    recursively rolls a single t0 forward) predicts every t0 present in its
    input in one batch, from the model's analytic mean/std (no sampling),
    unscaled back to each target's raw units.

    Uses a model with a fixed, known scaled mean/std and dataset_scaled's
    plain DiffScaler target (no chaining) so the expected raw output can be
    hand-computed: DiffScaler's inverse is exactly "last known actual value
    + predicted diff", so the unscaled mean should be last-known-value +
    scaled_mean, and the unscaled std should equal scaled_std unchanged
    (DiffScaler's inverse is a pure +anchor shift, so it doesn't rescale
    magnitude).
    """
    import numpy as np
    from futureish.forecasting.inference import InferenceEngine
    from futureish.modeling.base import ABCModel

    scaled_mean = 2.5
    scaled_std = 0.75

    class FixedModel(ABCModel):
        def _fit(self, X, features, config=None):
            pass

        def _predict(self, X):
            return {name: (np.full(X.shape[0], scaled_mean), np.full(X.shape[0], scaled_std)) for name in self.feature_names}

        def _predict_mean(self, X):
            return {name: mean for name, (mean, _) in self._predict(X).items()}

        def _predict_sample(self, X):
            return {name: mean for name, (mean, _) in self._predict(X).items()}

    forecaster = Forecaster(FixedModel())
    forecaster.fit(dataset_scaled)

    inference_model = InferenceEngine(forecaster)

    mark_times = [datetime(2023, 1, 1, 0, m, 0) for m in (3, 5, 7)]
    data_t0 = basic_data.with_columns(
        pl.when(pl.col("time").is_in(mark_times)).then(pl.col("time")).otherwise(None).alias("t0")
    )

    pred = inference_model.predict_single_step(data_t0, with_bounds=True).sort("t0")

    expected = (
        basic_data
        .filter(pl.col("time").is_in(mark_times))
        .select(
            pl.col("time").alias("t0"),
            pl.col("time").dt.offset_by("1m"),
            (pl.col("value_c") + scaled_mean).alias("value_c_mean"),
            pl.lit(scaled_std).alias("value_c_std"),
        )
        .sort("t0")
    )

    assert_frame_equal(pred.select("t0", "time", "value_c_mean", "value_c_std"), expected)

@pytest.fixture(params=[True, False])
def online_dataset(basic_data: pl.DataFrame, request: pytest.FixtureRequest) -> OnlineDataset:
    from futureish.datasets.online import OnlineDataset
    from futureish.features.recorded import RecordedInputFeature
    from futureish.datapoints.recorded import RecordedDataPoint
    from futureish.features.target import TargetFeature
    from futureish.scaling import ZScoreScaler, DiffScaler

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")
    dp_c = RecordedDataPoint(name="value_c", known_future=True, step_size="1m")

    # Define input features
    input_features: list[InputFeatureBase] = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m", scaler_factory=ZScoreScaler),
        RecordedInputFeature(base_datapoint=dp_b, feature_name="B", uses_future_data=False, input_length="1m", scaler_factory=ZScoreScaler),
    ]
    if request.param:
        input_features.append(
            SinCosTimeFeature(second=True, minute=True, hour=True)
        )

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C", scaler_factory=DiffScaler),
    ]

    # Create dataset
    dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    
    return dataset

from futureish.datasets.online import OnlineDataset

@pytest.mark.parametrize("n_samples", [1, 2])
def test_recursive_forecast_dummy_all_trajs(online_dataset: OnlineDataset, n_samples: int):
    from futureish._internal.dummy import DummyModel

    online_training_dataset, prediction_dataset = online_dataset.split_training_and_test_data("3m")
    forecaster = Forecaster(DummyModel())
    forecaster.fit(online_training_dataset)

    pred_trajs = prediction_dataset.create_all_trajectories("3m")

    pred = forecaster.predict_autoregressive(pred_trajs, samples=n_samples)

    assert pred.height > 0, "Predictions should have non-zero height"
    assert pred.select(pl.max_horizontal(pl.exclude("t0").null_count())).item() == 0, "Predictions should not contain null values"