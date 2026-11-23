import pytest
import polars as pl
from datetime import datetime, timedelta
from futureish.datapoints import RecordedDataPoint
from futureish.features import RecordedInputFeature, TargetFeature
from futureish.scaling import ZScoreScaler, ChainedScaler, DiffScaler
from futureish.datasets import Dataset
from futureish.forecasting import Forecaster
from futureish._internal.dummy import DummyModel

@pytest.fixture
def sparse_data():
    sparse = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2023, 1, 1, 0, 0, 0),
                end=datetime(2023, 12, 31, 23, 59, 59),
                interval="1d",
                eager=True
            )
        }
    ).with_columns(
        pl.int_range(pl.len()).alias("value")
    ).with_columns(
        pl.when(pl.col("value") % 2 == 0)
        .then(pl.col("value"))
        .otherwise(None)
        .alias("sparse_value")
    )

    return sparse

@pytest.fixture
def sparse_dataset(sparse_data):
    dp = RecordedDataPoint(name="value", step_size="1d", known_future=True)
    dp_sparse = RecordedDataPoint(name="sparse_value", step_size="2d")

    input_features = [
        RecordedInputFeature(dp, feature_name="value_input", input_length="5d", scaler_factory=ZScoreScaler, debug=True),
        RecordedInputFeature(dp_sparse, feature_name="sparse_value_input", input_length="5d", scaler_factory=ZScoreScaler, debug=True),
    ]
    target_features = [
        TargetFeature(dp_sparse, feature_name="sparse_value_target", scaler_factory=ChainedScaler.from_scalers(DiffScaler, ZScoreScaler), debug=True),
    ]
    dataset = Dataset(
        sparse_data,
        input_features=input_features,
        target_features=target_features,
    )
    return dataset

@pytest.mark.filterwarnings("ignore:sparse_value with step size 2d does not align with the interval of the data")
def test_sparse_data(sparse_dataset: Dataset):
    sparse_dataset.to_train_dataset().create_training_data()

    model = Forecaster(DummyModel())
    model.fit(sparse_dataset.to_train_dataset())

    traj, _ = sparse_dataset.split_training_and_test_data(0.1)[1].create_prediction_trajectory(pred_len="10d")

    model.predict_autoregressive(traj)

@pytest.fixture
def sparse_data2():
    import polars as pl
    from datetime import datetime, timedelta
    data = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 12, 31),
            interval="1d",
            eager=True,
        )
    }).with_columns(
         (pl.int_range(pl.len()) // 2).alias("dense_repeating") * 1.1,
         pl.when(pl.row_index() % 2 == 0).then(pl.int_range(pl.len())).alias("sparse"),
    )
    return data

@pytest.mark.filterwarnings("ignore:.*:UserWarning")
def test_repr(sparse_data2: pl.DataFrame):
    from futureish.datapoints import RecordedDataPoint
    from futureish.features import TargetFeature, RecordedInputFeature, SinCosTimeFeature, BucketInputFeature
    from futureish.scaling import DiffScaler, ZScoreScaler, ChainedScaler
    from futureish.datasets import OnlineTrainingDataset, Dataset

    dp_sparse = RecordedDataPoint(name="sparse", step_size="2d")
    dp_dense = RecordedDataPoint(name="dense_repeating", step_size="2d")

    input_features = [
        SinCosTimeFeature(ordinal_day=True)
    ]
    target_features = [
        TargetFeature(dp_sparse, feature_name="o2_line1_tank1_target", debug=True),
        TargetFeature(dp_dense, feature_name="inflow_line1_target", scaler_factory=DiffScaler, debug=True),
    ]

    dataset = Dataset(sparse_data2, input_features=input_features, target_features=target_features)

    dataset_train, dataset_test = dataset.split_training_and_test_data(0.8)

    unique_values = dataset_train.create_training_data().get_column("inflow_line1_target").unique().len()
    assert unique_values != 1, "error: inflow_line1_target is constant in training data, which is unexpected"