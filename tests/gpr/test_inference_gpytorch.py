from typing import cast

import polars as pl
from datetime import datetime
from polars.testing import assert_frame_equal
from pytest import fixture
import pytest
import torch

from futureish.datasets.train import TrainDataset
pytest.importorskip("gpytorch")

from futureish.datasets.online import OnlineDataset
from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchFitConfig

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
    from futureish.scaling import DiffScaler
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C", scaler_factory=DiffScaler),
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
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C", scaler_factory=ZScoreScaler),
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
    train_dataset, _ = online_dataset.split_training_and_test_data(basic_data_zscale.height + 1)

    return train_dataset

@pytest.mark.slow
def test_recursive_forecast_gpr_gpytorch(dataset: TrainDataset, basic_data: pl.DataFrame):
    from futureish.forecasting.forecaster import Forecaster
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer
    import gpytorch
    import torch

    def kernel_factory(batch_shape: torch.Size, nd: int, x: torch.Tensor, y: torch.Tensor):
        kern = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=nd, batch_shape=batch_shape)
        return gpytorch.kernels.ScaleKernel(kern, batch_shape=batch_shape)

    forecaster = Forecaster(GPyTorchTrainer(induce_size=5, kernel_factory=kernel_factory, use_cuda=torch.cuda.is_available()))
    forecaster.fit(dataset, config=GPyTorchFitConfig(iters=100000))

    dummy_data_prediction = (
        basic_data
            .select(
                "time",
                pl.col("value_a"),
                pl.col("value_b"),
                pl.when(pl.col("time") < datetime(2023, 1, 1, 0, 3, 0)).then("value_c").otherwise(None),
                pl.when(pl.col("time") == datetime(2023, 1, 1, 0, 2, 0)).then("time").otherwise(None).alias("t0"),
            )
    )

    pred = forecaster.predict_autoregressive(dummy_data_prediction)

    expected = basic_data.select(sorted(dummy_data_prediction.drop("t0").columns)).with_columns()

    pred = pred.select(sorted(pred.columns)).with_columns(pl.col("value_c").round()).drop("t0")

    assert_frame_equal(pred, expected, check_dtypes=False)

@pytest.mark.slow
def test_recursive_forecast_gpr_gpytorch_scaled(dataset_scaled: TrainDataset, basic_data: pl.DataFrame):
    from futureish.forecasting.forecaster import Forecaster
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer
    import gpytorch
    import torch

    ard_num_dims = dataset_scaled.create_training_data().select(pl.col("X").arr.len().first()).item()

    def kernel_factory(batch_shape: torch.Size, nd: int, x: torch.Tensor, y: torch.Tensor):
        kern = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
        return gpytorch.kernels.ScaleKernel(kern, batch_shape=batch_shape)

    forecaster = Forecaster(GPyTorchTrainer(induce_size=5, kernel_factory=kernel_factory, use_cuda=torch.cuda.is_available()))
    forecaster.fit(dataset_scaled, config=GPyTorchFitConfig(iters=10000))

    dummy_data_prediction = (
        basic_data
            .select(
                "time",
                pl.col("value_a"),
                pl.col("value_b"),
                pl.when(pl.col("time") < datetime(2023, 1, 1, 0, 3, 0)).then("value_c").otherwise(None),
                pl.when(pl.col("time") == datetime(2023, 1, 1, 0, 2, 0)).then("time").otherwise(None).alias("t0"),
            )
    )

    pred = forecaster.predict_autoregressive(dummy_data_prediction)

    expected = basic_data.select(sorted(dummy_data_prediction.drop("t0").columns)).with_columns()

    pred = pred.select(sorted(pred.columns)).with_columns(pl.col("value_c").round()).drop("t0")

    assert_frame_equal(pred, expected, check_dtypes=False)