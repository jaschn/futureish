import polars as pl
from datetime import datetime
from pytest import fixture
import pytest

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
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C"),
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

@pytest.mark.parametrize("n_samples", [1, 2])
def test_recursive_gpytorch_model(dataset: TrainDataset, basic_data: pl.DataFrame, n_samples: int):
    from futureish.forecasting.forecaster import Forecaster
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer
    import gpytorch

    def kernel_factory(batch_shape, x_dims, X, Y):
        kern = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=x_dims, batch_shape=batch_shape)
        return gpytorch.kernels.ScaleKernel(kern, batch_shape=batch_shape)

    forecaster = Forecaster(GPyTorchTrainer(induce_size=5, kernel_factory=kernel_factory))
    forecaster.fit(dataset, config=GPyTorchFitConfig(iters=2))

    t0 = datetime(2023, 1, 1, 0, 3, 0)

    dummy_data_prediction = (
        basic_data
            .select(
                "time",
                pl.col("value_a"),
                pl.col("value_b"),
                pl.when(pl.col("time") < t0).then("value_c").otherwise(None),
                pl.when(pl.col("time").dt.offset_by("1m") == t0).then("time").otherwise(None).alias("t0")
            )
    )

    pred = forecaster.predict_autoregressive(dummy_data_prediction, n_samples)

    assert pred.height > 0, "No data predicted"
    assert pred.select(pl.all_horizontal(pl.exclude("t0").is_not_null().all())).item(), "Some predictions are null"