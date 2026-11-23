#type: ignore

import itertools
import re

import polars as pl
from datetime import datetime, timedelta
from pytest import fixture
import pytest

from futureish.datasets.base import DatasetBase
from futureish.features.target import TargetFeature
from futureish.datasets.predict import PredictDatasetBase
from futureish.datasets.train import TrainDataset
from futureish.scaling import DiffScaler, ZScoreScaler, NoopScaler, ScalerBase
from futureish.features.recorded import RecordedInputFeature
from futureish.datapoints.recorded import RecordedDataPoint
from futureish.datasets.online import OnlineDataset
from futureish.datasets.config import DataConfig
from futureish._internal.constants import CONTIGUOUS_GROUP

from polars.testing import assert_frame_equal, assert_series_equal


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
            "value_d": pl.arange(30, 41, eager=True),
            "value_e": pl.arange(40, 51, eager=True),
        }
    )
    return dummy_data

@fixture
def data_with_gap() -> pl.DataFrame:
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
            "value_d": pl.arange(30, 41, eager=True),
            "value_e": pl.arange(40, 51, eager=True),
        }
    )
    return dummy_data.filter(pl.col("time") != datetime(2023, 1, 1, 0, 5, 0))

@fixture
def long_data_with_gaps() -> pl.DataFrame:
    data_len = 1000
    gap_cnt = 10
    data_start = datetime(2023, 1, 1, 0, 0, 0)
    data_end = data_start + timedelta(minutes=data_len - 1)

    dummy_data = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=data_start,
                end=data_end,
                interval="1m",
                eager=True,
            ),
            "value_a": pl.arange(0, data_len , eager=True),
            "value_b": pl.arange(data_len, data_len*2, eager=True),
        }
    )
    gap_idxs = range(0, data_len, data_len // gap_cnt)
    gaps: list[datetime] = []
    for i, idx in enumerate(gap_idxs):
        gaps_start = dummy_data[idx, "time"]
        gap = pl.datetime_range(
            start=gaps_start,
            end=gaps_start + timedelta(minutes=i),
            interval="1m",
            eager=True,
        ).to_list()
        gaps.extend(gap)

    dummy_data_with_gaps = (
        dummy_data
        .filter(
            ~pl.col("time").is_in(gaps)
        )
    )

    return dummy_data_with_gaps

@fixture
def basic_online_dataset(basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
    ]
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="B"),
    ]

    return OnlineDataset(
        raw_data=basic_data,
        input_features=input_features,
        target_features=target_features,
        size_of_first_online_window="1m",
    )


def test_data_config_alias_and_validation(basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    config = DataConfig(
        input_features=[RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m")],
        target_features=[TargetFeature(base_datapoint=dp_b, feature_name="B")],
        time_column_dtype=pl.Datetime(time_unit="ms", time_zone="UTC"),
    )
    assert config.longest_input_length == "1m"
    assert config.shortest_input_length == "1m"

    online_dataset = OnlineDataset(
        raw_data=basic_data,
        input_features=[RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m")],
        target_features=[TargetFeature(base_datapoint=dp_b, feature_name="B")],
        size_of_first_online_window="1m",
    )
    with pytest.raises(ValueError, match="split must be between 0 and 1"):
        online_dataset.split_training_and_test_data(-0.1)


def test_invalid_input_feature_raises_value_error():
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    with pytest.raises(ValueError, match="Input length must be a positive duration"):
        RecordedInputFeature(
            base_datapoint=dp_a,
            feature_name="A",
            input_length="-1m",
            uses_future_data=False,
        )

    with pytest.raises(ValueError, match="Feature name must be different from base datapoint name"):
        RecordedInputFeature(
            base_datapoint=dp_a,
            feature_name="value_a",
            input_length="1m",
            uses_future_data=False,
        )


def test_dataset_initialization(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="5m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="B"),
    ]

    # Create dataset
    dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    assert dataset.data is not None
    assert len(dataset.config.input_features) == 1
    assert len(dataset.config.target_features) == 1
    assert "time" in dataset.data.columns
    assert CONTIGUOUS_GROUP in dataset.data_with_contiguous_group().collect_schema().names()
    assert dataset.config.longest_input_length == "5m"

def test_dataset_handles_gaps(data_with_gap: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="3m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="B"),
    ]

    # Create dataset
    dataset = OnlineDataset(raw_data=data_with_gap, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    assert dataset.data_with_contiguous_group().select(pl.col(CONTIGUOUS_GROUP).n_unique()).collect().item() == 2

    groups = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2023, 1, 1, 0, 0, 0),
                end=datetime(2023, 1, 1, 0, 10, 0),
                interval="1m",
                eager=True,
            ),
            CONTIGUOUS_GROUP: [0] * 5 + [None] + [1] * 5
        }
    )
    groups =  groups.filter(pl.col("time") != datetime(2023, 1, 1, 0, 5, 0))

    assert dataset.data_with_contiguous_group().select(pl.col("time"), pl.col(CONTIGUOUS_GROUP)).collect().equals(groups)

def test_dataset_longest_input_length(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
        RecordedInputFeature(base_datapoint=dp_a, feature_name="B", uses_future_data=False, input_length="3m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="C"),
    ]

    # Create dataset
    dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    assert dataset.config.longest_input_length == "3m"


def test_dataset_data_splits(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="B"),
    ]

    # Create dataset and split it into a train/predict pair
    online_dataset = OnlineDataset(
        raw_data=basic_data,
        input_features=input_features,
        target_features=target_features,
        size_of_first_online_window="1m",
    )
    total_valid_t0 = online_dataset.valid_t0.height

    train_dataset, predict_dataset = online_dataset.split_training_and_test_data(0.6)

    assert train_dataset.valid_t0.height == 6
    assert predict_dataset.valid_t0.height == 4
    assert train_dataset.valid_t0.height + predict_dataset.valid_t0.height == total_valid_t0



def test_dataset_calc_t0(data_with_gap: pl.DataFrame):
    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
        RecordedInputFeature(base_datapoint=dp_a, feature_name="B", uses_future_data=False, input_length="2m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="C"),
    ]

    # Create dataset
    dataset = OnlineDataset(raw_data=data_with_gap, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    assert dataset.config.longest_input_length == "2m"
    assert dataset.config.shortest_input_length == "1m"
    assert dataset.config.step_size == "1m"
    
    assert dataset.valid_t0.height == 6
    expected = pl.DataFrame(
        {
            "t0": [
                datetime(2023, 1, 1, 0, 2, 0),
                datetime(2023, 1, 1, 0, 3, 0),
                datetime(2023, 1, 1, 0, 4, 0),
                datetime(2023, 1, 1, 0, 8, 0),
                datetime(2023, 1, 1, 0, 9, 0),
                datetime(2023, 1, 1, 0, 10, 0),
            ]
        }
    )
    assert_frame_equal(dataset.valid_t0, expected)

def test_dont_allow_overlapping_feature_names(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=False, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features with overlapping names
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="B"),
    ]

    # Attempt to create dataset and expect an error
    with pytest.raises(ValueError, match="Feature names must be unique"):
        OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")


def test_dataset_create_training_data(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
        RecordedInputFeature(base_datapoint=dp_b, feature_name="B", uses_future_data=False, input_length="2m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="C"),
    ]
    # Create dataset and use the training subdataset from a 60% split
    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    train_dataset, _ = online_dataset.split_training_and_test_data(0.6)

    training_data = train_dataset.create_training_data()

    expected = pl.DataFrame(
        {
            "t0": [
                    datetime(2023, 1, 1, 0, 2, 0),
                    datetime(2023, 1, 1, 0, 3, 0),
                    datetime(2023, 1, 1, 0, 4, 0),
                    datetime(2023, 1, 1, 0, 5, 0),
                    datetime(2023, 1, 1, 0, 6, 0),
            ],
            "X": [
                [1.0, 2.0, 10.0, 11.0, 12.0],
                [2.0, 3.0, 11.0, 12.0, 13.0],
                [3.0, 4.0, 12.0, 13.0, 14.0],
                [4.0, 5.0, 13.0, 14.0, 15.0],
                [5.0, 6.0, 14.0, 15.0, 16.0],
            ]
        },
    ).with_columns(pl.col("X").list.to_array(5))

    assert_frame_equal(training_data.select("t0", "X").sort("t0"), expected)

def test_dataset_scaling(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m", scaler_factory=ZScoreScaler),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="C"),
    ]

    # Create dataset. The online window and split are both sized to cover
    # exactly the timestamps the training subdataset's X windows touch, so
    # the ZScoreScaler's fit stats are self-consistent with what's measured
    # below (see test_dataset_0m_input_scaling for the same trick).
    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="8m")
    train_dataset, _ = online_dataset.split_training_and_test_data("8m")

    training_data = train_dataset.create_training_data()

    std = (
            training_data
            .select(
                pl.concat(
                    [
                        pl.col("X").arr.first(),
                        pl.col("X").last().arr.last(),
                    ]
                ).std()
            )
        )
    
    mean = (
        training_data
        .select(
            pl.col("X").arr.explode(empty_as_null=True).mean().alias("X_mean"),
        )
    )

    assert std.item() == pytest.approx(1.0, rel=1e-2)
    assert mean.select(pl.col("X_mean")).item() == pytest.approx(0.0, rel=1e-2)


def test_dataset_0m_input_scaling(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="0m", scaler_factory=ZScoreScaler),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="C"),
    ]

    # Create dataset. The online window ("8m") and split ("9m") are sized so
    # the ZScoreScaler is fit on exactly the timestamps the training
    # subdataset's X windows touch (t=0..8), matching what's measured below.
    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="8m")
    train_dataset, _ = online_dataset.split_training_and_test_data("8m")

    data = train_dataset.create_training_data().select(pl.col("X").list.explode(empty_as_null=True)).to_series()
    assert data.mean() == pytest.approx(0.0, rel=1e-2)
    assert data.std() == pytest.approx(1.0, rel=1e-2)


def test_dataset_future(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=False, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=True, input_length="1m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="C"),
    ]

    # Create dataset and use the training subdataset from a 60% split
    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    train_dataset, _ = online_dataset.split_training_and_test_data(0.6)

    training_data = train_dataset.create_training_data()
    expected = pl.DataFrame(
        {
            "t0": [
                datetime(2023, 1, 1, 0, 1, 0),
                datetime(2023, 1, 1, 0, 2, 0),
                datetime(2023, 1, 1, 0, 3, 0),
                datetime(2023, 1, 1, 0, 4, 0),
                datetime(2023, 1, 1, 0, 5, 0),
                datetime(2023, 1, 1, 0, 6, 0),
            ],
            "X": [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
                [5.0, 6.0, 7.0],
            ]
        }
    ).with_columns(pl.col("X").list.to_array(3))

    assert_frame_equal(training_data.select("t0", "X").sort("t0"), expected)

def test_dataset_feature(basic_data: pl.DataFrame):
    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")
    dp_c = RecordedDataPoint(name="value_c", known_future=True, step_size="1m")
    dp_d = RecordedDataPoint(name="value_d", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
        RecordedInputFeature(base_datapoint=dp_b, feature_name="B", uses_future_data=True, input_length="1m"),

    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C"),
        TargetFeature(base_datapoint=dp_d, feature_name="D")
    ]

    # Create dataset and use the training subdataset from a 60% split
    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    train_dataset, _ = online_dataset.split_training_and_test_data(0.6)

    target = pl.DataFrame(
        {
            "t0": [
                datetime(2023, 1, 1, 0, 1, 0),
                datetime(2023, 1, 1, 0, 2, 0),
                datetime(2023, 1, 1, 0, 3, 0),
                datetime(2023, 1, 1, 0, 4, 0),
                datetime(2023, 1, 1, 0, 5, 0),
                datetime(2023, 1, 1, 0, 6, 0),
            ],
            "C": [
                [22.0],
                [23.0],
                [24.0],
                [25.0],
                [26.0],
                [27.0],
            ],
            "D": [
                [32.0],
                [33.0],
                [34.0],
                [35.0],
                [36.0],
                [37.0],
            ]
        }
    ).with_columns(pl.col("C").list.to_array(1), pl.col("D").list.to_array(1))

    training_data = train_dataset.create_training_data()

    assert_frame_equal(training_data.select(pl.exclude("X")).sort("t0"), target)


def test_dataset_target_scaling(basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_c = RecordedDataPoint(name="value_c", known_future=True, step_size="1m")

    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
    ]

    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C", scaler_factory=DiffScaler),
    ]

    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    train_dataset, _ = online_dataset.split_training_and_test_data(0.6)

    expected = pl.DataFrame(
        {
            "t0": [
                datetime(2023, 1, 1, 0, 1, 0),
                datetime(2023, 1, 1, 0, 2, 0),
                datetime(2023, 1, 1, 0, 3, 0),
                datetime(2023, 1, 1, 0, 4, 0),
                datetime(2023, 1, 1, 0, 5, 0),
                datetime(2023, 1, 1, 0, 6, 0),
            ],
            "C": [
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
            ],
        }
    ).with_columns(pl.col("C").list.to_array(1))

    training_data = train_dataset.create_training_data()

    assert_frame_equal(
        training_data.select("t0", "C"),
        expected,
        check_row_order=False,
    )


def test_dataset_with_scaler(basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_c = RecordedDataPoint(name="value_c", known_future=True, step_size="1m")

    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="0m"),
    ]

    target_features = [
        TargetFeature(base_datapoint=dp_c, feature_name="C", scaler_factory=DiffScaler),
    ]

    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    train_dataset, _ = online_dataset.split_training_and_test_data(0.6)

    training_data = train_dataset.create_training_data()

    assert training_data.select(pl.all_horizontal(pl.all().is_not_null().all())).item()

@pytest.mark.parametrize("scaler_input", [NoopScaler, ZScoreScaler, DiffScaler])
@pytest.mark.parametrize("scaler_target", [NoopScaler, ZScoreScaler, DiffScaler])
@pytest.mark.parametrize("removed", [([3], 6), ([8], 6), ([3, 8], 3)])
def test_dataset_create_non_contigous_training_data(basic_data: pl.DataFrame, removed: tuple[list[int], int], scaler_input: type[ScalerBase], scaler_target: type[ScalerBase]):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m", scaler_factory=scaler_input),
        RecordedInputFeature(base_datapoint=dp_b, feature_name="B", uses_future_data=False, input_length="1m", scaler_factory=scaler_input),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="C", scaler_factory=scaler_target),
    ]

    idxs = list(range(basic_data.height))
    for i in sorted(removed[0], reverse=True):
        idxs.pop(i)

    non_contious = (
        basic_data
        .gather(idxs)
    )

    # Create dataset, sizing the online window to cover the whole (possibly
    # gapped) span so the scaler is fit over everything, and splitting past
    # the last row (clamped to it) so every valid t0 ends up in training.
    span_minutes = int((non_contious["time"].max() - non_contious["time"].min()).total_seconds() // 60)
    online_dataset = OnlineDataset(
        raw_data=non_contious,
        input_features=input_features,
        target_features=target_features,
        size_of_first_online_window=f"{span_minutes}m",
    )
    train_dataset, _ = online_dataset.split_training_and_test_data(non_contious.height + 1)

    training_data = train_dataset.create_training_data()

    assert training_data.height == removed[1]
    assert training_data.select(pl.all_horizontal(pl.all().is_not_null().all())).item()



def test_data_config_no_target_features_raises():
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    with pytest.raises(ValueError, match="No target features defined"):
        DataConfig(
            input_features=[RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m")],
            target_features=[],
            time_column_dtype=pl.Datetime(time_unit="us"),
        )


def test_data_config_duplicate_feature_names_raises():
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    with pytest.raises(ValueError, match="Feature names must be unique"):
        DataConfig(
            input_features=[RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m")],
            target_features=[TargetFeature(base_datapoint=dp_a, feature_name="A")],
            time_column_dtype=pl.Datetime(time_unit="us"),
        )


def test_create_input_duplicate_time_without_group_raises(basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    input_feature = RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m")
    config = DataConfig(
        input_features=[input_feature],
        target_features=[TargetFeature(base_datapoint=dp_b, feature_name="B")],
        time_column_dtype=basic_data.collect_schema()["time"],
    )

    duplicated = pl.concat([basic_data, basic_data]).with_columns(pl.when(pl.col("time") == pl.col("time").max()).then("time").alias("t0"))
    config.input_features = [input_feature.fitted(duplicated)]

    with pytest.raises(ValueError, match="Dataframe contains duplicate time values but no group column specified"):
        config.create_input(duplicated)


def test_create_xy_raises_when_no_data_points_overlap():
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    input_feature = RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="0m")
    target_feature = TargetFeature(base_datapoint=dp_b, feature_name="B")
    config = DataConfig(
        input_features=[input_feature],
        target_features=[target_feature],
        time_column_dtype=pl.Datetime(time_unit="us"),
    )

    # A single row means the target's time.shift(1) is null, so X (keyed by the real
    # timestamp) and Y (keyed by the shifted, null timestamp) can never join.
    single_row = pl.DataFrame({
        "time": [datetime(2023, 1, 1, 0, 0, 0)],
        "value_a": [0],
        "value_b": [10],
    })
    single_row = single_row.with_columns(pl.col("time").alias("t0"))
    config.input_features = [input_feature.fitted(single_row)]
    config.target_features = [target_feature.fitted(single_row)]

    with pytest.raises(ValueError, match="No data points found for training"):
        config.create_XY(single_row)


# --- Dataset column/consistency validation edge cases ---

def test_dataset_missing_input_base_column_raises(basic_data: pl.DataFrame):
    dp_missing = RecordedDataPoint(name="value_missing", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    with pytest.raises(ValueError, match=re.escape("Raw data is missing required feature columns: ['value_missing']")):
        OnlineDataset(
            raw_data=basic_data,
            input_features=[RecordedInputFeature(base_datapoint=dp_missing, feature_name="A", uses_future_data=False, input_length="1m")],
            target_features=[TargetFeature(base_datapoint=dp_b, feature_name="B")],
            size_of_first_online_window="1m",
        )


def test_dataset_missing_target_base_column_raises(basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_missing = RecordedDataPoint(name="value_missing", known_future=True, step_size="1m")

    with pytest.raises(ValueError, match=re.escape("Raw data is missing required feature columns: ['value_missing']")):
        OnlineDataset(
            raw_data=basic_data,
            input_features=[RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m")],
            target_features=[TargetFeature(base_datapoint=dp_missing, feature_name="B")],
            size_of_first_online_window="1m",
        )


def test_dataset_future_data_input_without_matching_target_raises(basic_data: pl.DataFrame):
    dp_a_not_future = RecordedDataPoint(name="value_a", known_future=False, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    with pytest.raises(ValueError, match="Input feature that uses future data must have a base datapoint with known future"):
        OnlineDataset(
            raw_data=basic_data,
            input_features=[RecordedInputFeature(base_datapoint=dp_a_not_future, feature_name="A", uses_future_data=True, input_length="1m")],
            target_features=[TargetFeature(base_datapoint=dp_b, feature_name="B")],
            size_of_first_online_window="1m",
        )


# --- InputFeature / FeaturesBase validation edge cases ---

def test_input_feature_requires_known_future_for_future_data():
    dp_not_future = RecordedDataPoint(name="value_a", known_future=False, step_size="1m")

    with pytest.raises(ValueError, match="Input feature that uses future data must have a base datapoint with known future"):
        RecordedInputFeature(base_datapoint=dp_not_future, feature_name="A", uses_future_data=True, input_length="1m")


def test_target_feature_name_collision_with_base_datapoint_raises():
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    with pytest.raises(ValueError, match="Feature name must be different from base datapoint name"):
        TargetFeature(base_datapoint=dp_a, feature_name="value_a")


def test_features_base_fitted_is_reusable_and_does_not_mutate_original():
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    feature = TargetFeature(base_datapoint=dp_a, feature_name="C")
    df = pl.DataFrame({"value_a": [1.0, 2.0, 3.0]})

    fitted_once = feature.fitted(df)
    fitted_twice = feature.fitted(df)

    assert fitted_once is not fitted_twice
    assert fitted_once.scaler is not fitted_twice.scaler
    with pytest.raises(ValueError, match="Scaler has not been initialized"):
        _ = feature.scaler


def test_features_base_scaler_access_before_init_raises():
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    feature = TargetFeature(base_datapoint=dp_a, feature_name="C")

    with pytest.raises(ValueError, match="Scaler has not been initialized"):
        _ = feature.scaler


# --- TargetFeature.past_predict_step edge cases ---

def test_target_feature_past_predict_step_clips_to_bounds():
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")
    feature = TargetFeature(base_datapoint=dp_b, feature_name="C", bounds_min=0.0, bounds_max=5.0)

    df = pl.DataFrame({"value_b": [-2.0, 1.0, 2.0, 10.0]})
    clipped = feature.past_predict_step(df)

    assert clipped["value_b"].to_list() == [0.0, 1.0, 2.0, 5.0]


def test_target_feature_past_predict_step_noop_without_bounds():
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")
    feature = TargetFeature(base_datapoint=dp_b, feature_name="C")

    df = pl.DataFrame({"value_b": [-2.0, 1.0, 2.0, 10.0]})
    assert feature.past_predict_step(df).equals(df)

@pytest.mark.parametrize("data_fixture", ["basic_data", "data_with_gap"])
def test_online_dataset(data_fixture: str, request: pytest.FixtureRequest):
    from futureish.datasets.online import OnlineDataset
    from datetime import timedelta

    data = request.getfixturevalue(data_fixture)

    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    input_features = [RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="0m")]
    target_features = [TargetFeature(base_datapoint=dp_a, feature_name="a")]

    online_dataset = OnlineDataset(raw_data=data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    online_training_dataset, prediction_dataset = online_dataset.split_training_and_test_data(0.2)
    online_training_dataset, prediction_dataset = online_dataset.split_training_and_test_data(timedelta(minutes=3))
    online_training_dataset, prediction_dataset = online_dataset.split_training_and_test_data("3m")

    def iter_test(dataset: DatasetBase):
        assert dataset.data.height > 0, "Expected non-empty prediction data, got empty"
        assert dataset.valid_t0.height > 0, "Expected non-empty valid_t0 in prediction dataset, got empty"
        training_valid_t0_len = dataset.valid_t0.height
        t0_cnt = 0
        splits = list(dataset.iter("2m"))

        for s in splits:
            assert s.data.height > 0, "Expected non-empty split data, got empty"
            assert s.valid_t0.height > 0, "Expected non-empty valid_t0, got empty"
            t0_cnt += s.valid_t0.height
            if isinstance(s, TrainDataset):
                assert s.create_training_data().height > 0, "Expected non-empty training data, got empty"
            if isinstance(s, PredictDatasetBase):
                assert s.create_prediction_trajectory(pred_len="0m")[0].height > 0, "Expected non-empty prediction trajectory, got empty"
                assert s.create_all_trajectories("0m").height > 0, "Expected non-empty all trajectories, got empty"
                assert s.create_all_single_points().height > 0, "Expected non-empty all single points, got empty"

        assert t0_cnt == training_valid_t0_len, "Expected total valid_t0 count to match original dataset, got {} vs {}".format(t0_cnt, training_valid_t0_len)

    iter_test(online_training_dataset)
    iter_test(prediction_dataset)

@pytest.mark.parametrize("seed", [42, 123, 456])
def test_random_subset(long_data_with_gaps: pl.DataFrame, seed: int):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    input_features = [RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="0m")]
    target_features = [TargetFeature(base_datapoint=dp_a, feature_name="a")]

    online_dataset = OnlineDataset(raw_data=long_data_with_gaps, input_features=input_features, target_features=target_features, size_of_first_online_window="2m")
    train_dataset, predict_dataset = online_dataset.split_training_and_test_data(0.5)

    train_subset = train_dataset.random_subset(50, random_seed=seed)
    assert train_subset.valid_t0.height == 50, f"Expected 50 valid_t0 in random subset, got {train_subset.valid_t0.height}"
    assert train_subset.create_training_data().height == 50, f"Expected 50 rows in training data for random subset, got {train_subset.create_training_data().height}"
    train_subset2 = train_subset.random_subset(20, random_seed=seed)
    assert train_subset2.valid_t0.height == 20, f"Expected 20 valid_t0 in random subset, got {train_subset2.valid_t0.height}"
    assert train_subset2.create_training_data().height == 20, f"Expected 20 rows in training data for random subset, got {train_subset2.create_training_data().height}"

    pred_subset = predict_dataset.random_subset(50, random_seed=seed, pred_len=predict_dataset.config.step_size.to_string())
    assert pred_subset.valid_t0.height == 50, f"Expected 50 valid_t0 in random subset, got {pred_subset.valid_t0.height}"
    assert pred_subset.create_all_trajectories(pred_subset.config.step_size.to_string()).select(pl.col("traj_id").n_unique()).item() == 50, f"Expected 50 unique trajectory IDs in random subset, got {pred_subset.create_all_trajectories(pred_subset.config.step_size.to_string()).select(pl.col("traj_id").n_unique()).item()}"
    pred_subset2 = pred_subset.random_subset(20, random_seed=seed)
    assert pred_subset2.valid_t0.height == 20, f"Expected 20 valid_t0 in random subset, got {pred_subset2.valid_t0.height}"
    assert pred_subset2.create_all_trajectories(pred_subset2.config.step_size.to_string()).select(pl.col("traj_id").n_unique()).item() == 20, f"Expected 20 unique trajectory IDs in random subset, got {pred_subset2.create_all_trajectories(pred_subset2.config.step_size.to_string()).select(pl.col('traj_id').n_unique()).item()}"

def test_sparse_input_features_basic():
    from futureish.features.sparse import SparseInputFeature
    data = pl.DataFrame({
        "time": pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1, 0, 59), interval="1m", eager=True),
        "value_a": [i for i in range(60)],
    })

    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    input_features = [
        SparseInputFeature(base_datapoint=dp_a, feature_name="A", input_length="5m", uses_future_data=False, reduction_factor=10),
    ]

    input_features_ref = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", input_length="0m", uses_future_data=False),
    ]

    target_features = [
        TargetFeature(base_datapoint=dp_a, feature_name="B"),
    ]

    dataset = OnlineDataset(raw_data=data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    dataset_ref = OnlineDataset(raw_data=data, input_features=input_features_ref, target_features=target_features, size_of_first_online_window="1m")

    dataset_training, prediction_dataset = dataset.split_training_and_test_data(0.5)
    dataset_training_ref, prediction_dataset_ref = dataset_ref.split_training_and_test_data(0.5)

    training_data = dataset_training.create_training_data()
    training_data_ref = dataset_training_ref.create_training_data()

    training_data_ref = training_data_ref.filter(pl.col("t0").is_in(training_data.get_column("t0")))

    assert_frame_equal(training_data_ref, training_data)


@pytest.mark.parametrize("input_len", [1, 2, 3, 4, 5, 6, 7, 10, 15, 20])
@pytest.mark.parametrize("reduction_factor", [1, 2, 3, 4, 5, 6])
def test_sparse_input_features(input_len: int, reduction_factor: int):
    from futureish.features.sparse import SparseInputFeature
    data = pl.DataFrame({
        "time": pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1, 0, 59), interval="1m", eager=True),
        "value_a": [i for i in range(60)],
    })

    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    expected_x_cnt = 1 + input_len // reduction_factor
    input_features = [
        SparseInputFeature(base_datapoint=dp_a, feature_name="A", input_length=f"{input_len}m", uses_future_data=False, reduction_factor=reduction_factor),
    ]

    target_features = [
        TargetFeature(base_datapoint=dp_a, feature_name="B"),
    ]

    dataset = OnlineDataset(raw_data=data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    dataset_training, prediction_dataset = dataset.split_training_and_test_data(0.5)

    training_data = dataset_training.create_training_data()

    assert training_data.get_column("X").arr.len().first() == expected_x_cnt, f"Expected {expected_x_cnt} elements in X, got {training_data.get_column('X').arr.len().first()}"

    comp = training_data.get_column("X").arr.sort() == training_data.get_column("X")

    assert comp.all(), "Expected X to be in order, but it was not"

    correct_diff = training_data.get_column("X").arr.eval((pl.element().diff().drop_nulls() == reduction_factor).all(), as_list=True).list.all().all()

    assert correct_diff, "Expected X to have correct differences, but it did not"

    pred_data = prediction_dataset.create_all_single_inputs()

    assert pred_data.get_column("X").arr.len().first() == expected_x_cnt, f"Expected {expected_x_cnt} elements in X, got {pred_data.get_column('X').arr.len().first()}"

    comp_pred = pred_data.get_column("X").arr.sort() == pred_data.get_column("X")

    assert comp_pred.all(), "Expected X to be in order, but it was not"

    correct_diff_pred = pred_data.get_column("X").arr.eval((pl.element().diff().drop_nulls() == reduction_factor).all(), as_list=True).list.all().all()

    assert correct_diff_pred, "Expected X to have correct differences, but it did not"

def test_modular_expressions_input_features():
    from futureish.features.modular import ModularInputFeature

    data = pl.DataFrame({
        "time": pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1, 0, 59), interval="1m", eager=True),
        "value_a": [i for i in range(60)],
    })

    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    def modification_expression(x: pl.DataFrame, feature_time: str, feature: str, grouped: list[str]) -> pl.DataFrame:
        x = (
            x
            .group_by(grouped)
            .agg(
                pl.col(feature).filter(pl.col(feature_time) == pl.col("t0")).first()
            )
        )
        return x

    def modification_expression2(x: pl.DataFrame, feature_time: str, feature: str, grouped: list[str]) -> pl.DataFrame:
        x = (
            x
            .group_by(grouped)
            .agg(
                pl.col(feature).sort_by(feature_time).gather([-2, -1])
            )
        )
        return x

    input_features_modular = [
        ModularInputFeature(base_datapoint=dp_a, feature_name="A", input_length=f"5m", uses_future_data=False, modification_expression=modification_expression),
        ModularInputFeature(base_datapoint=dp_a, feature_name="B", input_length=f"5m", uses_future_data=False, modification_expression=modification_expression2),
    ]

    input_features_ref = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", input_length=f"0m", uses_future_data=False),
        RecordedInputFeature(base_datapoint=dp_a, feature_name="B", input_length=f"1m", uses_future_data=False),
    ]

    target_features = [
        TargetFeature(base_datapoint=dp_a, feature_name="C"),
    ]

    dataset_modular = OnlineDataset(raw_data=data, input_features=input_features_modular, target_features=target_features, size_of_first_online_window="1m")

    dataset_ref = OnlineDataset(raw_data=data, input_features=input_features_ref, target_features=target_features, size_of_first_online_window="1m")
    
    dataset_training_modular, prediction_dataset_modular = dataset_modular.split_training_and_test_data(0.5)
    dataset_training_ref, prediction_dataset_ref = dataset_ref.split_training_and_test_data(0.5)

    training_data_modular = dataset_training_modular.create_training_data()
    training_data_ref = dataset_training_ref.create_training_data()

    training_data_ref = training_data_ref.filter(pl.col("t0") >= training_data_modular.get_column("t0").min())

    assert_frame_equal(training_data_ref, training_data_modular)


def test_bucket_expressions_input_features():
    from futureish.features.modular import ModularInputFeature
    from futureish.features.bucket import BucketInputFeature

    data = pl.DataFrame({
        "time": pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1, 0, 59), interval="1m", eager=True),
        "value_a": [i for i in range(60)],
    })

    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    def agg_expr(x: pl.Expr, feature_time: str) -> pl.Expr:
        return x.filter(pl.col(feature_time) == pl.col("t0")).first()

    def agg_expr2(x: pl.Expr, feature_time: str) -> pl.Expr:
        return x

    input_features_bucket = [
        BucketInputFeature(base_datapoint=dp_a, feature_name="A", input_length=f"4m", uses_future_data=False, group_by_window="5m", agg_expr=agg_expr),
        BucketInputFeature(base_datapoint=dp_a, feature_name="B", input_length=f"3m", uses_future_data=False, group_by_window="5m", agg_expr=agg_expr2),
    ]

    input_features_ref = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", input_length=f"0m", uses_future_data=False),
        RecordedInputFeature(base_datapoint=dp_a, feature_name="B", input_length=f"3m", uses_future_data=False),
    ]

    target_features = [
        TargetFeature(base_datapoint=dp_a, feature_name="C"),
    ]

    dataset_bucket = OnlineDataset(raw_data=data, input_features=input_features_bucket, target_features=target_features, size_of_first_online_window="1m")

    dataset_ref = OnlineDataset(raw_data=data, input_features=input_features_ref, target_features=target_features, size_of_first_online_window="1m")

    dataset_training_bucket, prediction_dataset_bucket = dataset_bucket.split_training_and_test_data(0.5)
    dataset_training_ref, prediction_dataset_ref = dataset_ref.split_training_and_test_data(0.5)

    training_data_bucket = dataset_training_bucket.create_training_data()
    training_data_ref = dataset_training_ref.create_training_data()

    training_data_ref = training_data_ref.filter(pl.col("t0") >= training_data_bucket.get_column("t0").min())

    assert_frame_equal(training_data_ref, training_data_bucket)

def test_dataset_prepare(basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
    ]
    target_features = [
        TargetFeature(base_datapoint=dp_a, feature_name="C"),
    ]

    online_dataset = OnlineDataset(raw_data=basic_data.lazy(), input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    with pytest.raises(ValueError, match="Raw data must contain a 'time' column"):
        OnlineDataset(raw_data=basic_data.drop("time"), input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    with pytest.raises(ValueError, match="Raw data must contain a 'time' column"):
        tmp = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
        tmp._prepare_raw_data(basic_data.drop("time")) # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ValueError, match=f"Raw data already contains '{CONTIGUOUS_GROUP}' column, which is reserved for internal use"):
        OnlineDataset(raw_data=basic_data.with_columns(pl.lit(1).alias(CONTIGUOUS_GROUP)), input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    with pytest.raises(ValueError, match="Raw data is empty"):
        OnlineDataset(raw_data=basic_data.filter(False), input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

def test_data_config_properties(basic_online_dataset: OnlineDataset):
    repr_fields = basic_online_dataset._repr_fields # pyright: ignore[reportPrivateUsage]
    for f in basic_online_dataset.__dict__:
        if f.startswith("_"):
            continue
        assert f in repr_fields, f"{f} not in _repr_fields"

common_feature_names = ["second", "minute", "hour", "weekday", "month", "ordinal_day"]
@pytest.mark.parametrize("feature_name", [list(x) for i in range(1, len(common_feature_names) + 1) for x in itertools.combinations(common_feature_names, i)])
def test_sin_cos_feature(basic_data: pl.DataFrame, feature_name: list[str]):
    from futureish.features.sincos_time import SinCosTimeFeature

    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

    input_features = [
        SinCosTimeFeature(**{f: True for f in feature_name})
    ]
    target_features = [
        TargetFeature(base_datapoint=dp_a, feature_name="A"),
    ]

    dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    training_dataset, _ = dataset.split_training_and_test_data(0.5)
    training_data = training_dataset.create_training_data()

    X = training_data.get_column("X")

    assert X.arr.len().first() == len(feature_name) * 2, f"Expected {len(input_features)} elements in X, got {X.arr.len().first()}"

    assert dataset.config.sub_features_names == [f"{f}_{s}" for f in feature_name for s in ["sin", "cos"]], f"Expected sub_features_names to be {[f'{f}_{s}' for f in feature_name for s in ['sin', 'cos']]}, got {dataset.config.sub_features_names}"

    assert X.arr.min().min() >= -1.0, f"Expected min of X to be >= -1.0, got {X.arr.min().min()}"
    assert X.arr.max().max() <= 1.0, f"Expected max of X to be <= 1.0, got {X.arr.max().max()}"

@pytest.mark.filterwarnings("ignore:b with step size 2m does not align with the interval of the data")
@pytest.mark.filterwarnings("ignore:c with step size 5m does not align with the interval of the data")
@pytest.mark.filterwarnings("ignore:d with step size 10m does not align with the interval of the data")
def test_dataset_with_irregular_input():
    dataset = (
        pl.DataFrame({
            "time": pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1, 1, 59), interval="1m", eager=True),
            "a": 0,
        }).with_columns(
            pl.when(pl.col("time").dt.minute() % 2 == 0).then(pl.lit(1)).alias("b"),
            pl.when(pl.col("time").dt.minute() % 5 == 0).then(pl.lit(2)).alias("c"),
            pl.when(pl.col("time").dt.minute() % 10 == 0).then(pl.lit(3)).alias("d"),
        )
    )

    dp_a = RecordedDataPoint(name="a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="b", known_future=True, step_size="2m")
    dp_c = RecordedDataPoint(name="c", known_future=True, step_size="5m")
    dp_d = RecordedDataPoint(name="d", known_future=True, step_size="10m")

    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="2m", debug=True),
        RecordedInputFeature(base_datapoint=dp_b, feature_name="B", uses_future_data=False, input_length="4m", debug=True),
        RecordedInputFeature(base_datapoint=dp_c, feature_name="C", uses_future_data=False, input_length="10m", debug=True),
    ]

    target_features = [
        TargetFeature(base_datapoint=dp_d, feature_name="D", debug=True),
    ]

    dataset = OnlineDataset(raw_data=dataset, input_features=input_features, target_features=target_features, size_of_first_online_window="20m")

    data_train, data_test = dataset.split_training_and_test_data(0.5)

    training_data = data_train.create_training_data()
    X = training_data.get_column("X")
    X_ref = pl.Series(
        "X",
        [
            [0.0] * 3 + [1.0] * 3 + [2.0] * 3
        ] * 6
    ).list.to_array(9)
    assert_series_equal(X, X_ref)

@pytest.mark.parametrize("split", [0.1, 0.5, 0.9, 1.0])
def test_dataset_train_predict_split(basic_online_dataset: OnlineDataset, split: float):
    basic_online_dataset.split(split)
    basic_online_dataset.split_training_and_test_data(split)

def test_dataset_split(basic_online_dataset: OnlineDataset):
    train, predict = basic_online_dataset.split_training_and_test_data(0.5)
    all_data = pl.concat([train.data, predict.data]).sort("time").unique()
    assert all_data.height == basic_online_dataset.data.height, f"Expected {basic_online_dataset.data.height} rows in split datasets, got {all_data.height}"

    train_all, _ = basic_online_dataset.split_training_and_test_data(1.0)
    all_data_iter = []
    for x in train_all.iter("3m"):
        all_data_iter.append(x.create_training_data())

    all_data_iter = pl.concat(all_data_iter).sort("t0")

    assert_frame_equal(all_data_iter, train_all.create_training_data().sort("t0"))

def test_dataset_iter(basic_online_dataset: OnlineDataset):
    for x in basic_online_dataset.iter("1m"):
        assert isinstance(x, OnlineDataset), f"Expected DatasetBase, got {type(x)}"

    train, predict = basic_online_dataset.split_training_and_test_data(0.5)
    for x in train.iter("1m"):
        assert isinstance(x, TrainDataset), f"Expected TrainDataset, got {type(x)}"

    for x in predict.iter("1m"):
        assert isinstance(x, PredictDatasetBase), f"Expected PredictDataset, got {type(x)}"

def test_dataset_splitting(basic_data: pl.DataFrame):

    # Define recorded data points
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    dp_b = RecordedDataPoint(name="value_b", known_future=True, step_size="1m")

    # Define input features
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A", uses_future_data=False, input_length="1m"),
    ]

    # Define target features
    target_features = [
        TargetFeature(base_datapoint=dp_b, feature_name="C"),
    ]

    online_dataset = OnlineDataset(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")

    train_dataset, prediction_dataset = online_dataset.split_training_and_test_data(0.5)

    all_t0 = pl.concat([train_dataset.valid_t0, prediction_dataset.valid_t0])
    assert not all_t0.get_column("t0").is_duplicated().any(), "Expected no duplicate t0 values across training and prediction datasets"

    assert_frame_equal(online_dataset.valid_t0, all_t0)

    train_split1, train_split2 = train_dataset.split(0.5)
    all_t0_train = pl.concat([train_split1.valid_t0, train_split2.valid_t0])
    assert not all_t0_train.get_column("t0").is_duplicated().any(), "Expected no duplicate t0 values across training splits"
    assert_frame_equal(train_dataset.valid_t0, all_t0_train)

    prediction_split1, prediction_split2 = prediction_dataset.split(0.5)
    all_t0_predict = pl.concat([prediction_split1.valid_t0, prediction_split2.valid_t0])
    assert not all_t0_predict.get_column("t0").is_duplicated().any(), "Expected no duplicate t0 values across prediction splits"
    assert_frame_equal(prediction_dataset.valid_t0, all_t0_predict)

from futureish.datasets import OnlineTrainingDataset
from futureish.datasets.forecast import ForecastingDataset
from futureish.datasets.test import TestDataset
from futureish.datasets.dataset import Dataset
@pytest.mark.parametrize("cls", [Dataset, OnlineTrainingDataset, OnlineDataset, TrainDataset, PredictDatasetBase, ForecastingDataset, TestDataset])
def test_all_classes(cls: type[DatasetBase], basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A"),
    ]
    target_features = [
        TargetFeature(base_datapoint=dp_a, feature_name="C"),
    ]
    if issubclass(cls, (OnlineTrainingDataset, OnlineDataset)):
        dataset = cls(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    elif issubclass(cls, (PredictDatasetBase, ForecastingDataset, TestDataset)):
        dataset_config = Dataset(raw_data=basic_data, input_features=input_features, target_features=target_features).config
        dataset = cls.create_with_config(raw_data=basic_data, config=dataset_config)
    else:
        dataset = cls(raw_data=basic_data, input_features=input_features, target_features=target_features)

    assert dataset.has_valid_t0(), f"Expected {cls.__name__} to have valid_t0, but it does not"
    dataset.data_with_valid_t0()

    dataset.split(0.5)
    list(dataset.iter("1m"))
    dataset.random_subset(2, random_seed=42)

@pytest.mark.parametrize("cls", [OnlineTrainingDataset, TrainDataset])
def test_train_datasets(cls: type[TrainDataset], basic_data: pl.DataFrame):
    dp_a = RecordedDataPoint(name="value_a", known_future=True, step_size="1m")
    input_features = [
        RecordedInputFeature(base_datapoint=dp_a, feature_name="A"),
    ]
    target_features = [
        TargetFeature(base_datapoint=dp_a, feature_name="C"),
    ]
    if "online" in cls.__name__.lower():
        dataset = cls(raw_data=basic_data, input_features=input_features, target_features=target_features, size_of_first_online_window="1m")
    else:
        dataset = cls(raw_data=basic_data, input_features=input_features, target_features=target_features)

    dataset.create_training_data()
    for x in dataset.iter("1m"):
        x.create_training_data()
