
from futureish.datapoints.time import TimeDataPoint
from futureish.datapoints.recorded import RecordedDataPoint
from futureish.datapoints.base import DataPoint
from futureish.features import BucketInputFeature, ModularInputFeature, SinCosTimeFeature, SparseInputFeature
from futureish.features.recorded import RecordedInputFeature
from futureish.features.target import TargetFeature
from futureish.modeling.base import ABCModel
from futureish._internal.repr import ReprMixin
import polars as pl
import pytest
from pytest import fixture
from datetime import datetime
from futureish.datasets import OnlineDataset

def create_basic_data() -> pl.DataFrame:
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

def basic_online_dataset():
    basic_data = create_basic_data()
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

def create_basic_data_point() -> RecordedDataPoint:
    return RecordedDataPoint(name="value_a", known_future=True, step_size="1m")

@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(c, id=f"{c.__class__.__name__}") for c in [
            DataPoint(name="value_b", known_future=True),
            RecordedDataPoint(name="value_a", known_future=True, step_size="1m"),
            TimeDataPoint(),
            basic_online_dataset().config,
            basic_online_dataset(),
            basic_online_dataset().split_training_and_test_data(0.5)[0],
            basic_online_dataset().split_training_and_test_data(0.5)[1],
            BucketInputFeature(base_datapoint=create_basic_data_point(), feature_name="A", uses_future_data=False, input_length="1m", group_by_window="1m", agg_expr=lambda exp, s: exp),
            ModularInputFeature(base_datapoint=create_basic_data_point(), feature_name="A", uses_future_data=False, input_length="1m", modification_expression=lambda df, a, b, c: df),
            RecordedInputFeature(base_datapoint=create_basic_data_point(), feature_name="A", uses_future_data=False, input_length="1m"),
            SinCosTimeFeature(second=True, minute=True, hour=True, weekday=True, ordinal_day=True, month=True),
            SparseInputFeature(base_datapoint=create_basic_data_point(), feature_name="A", uses_future_data=False, input_length="1m"),
            TargetFeature(base_datapoint=create_basic_data_point(), feature_name="A"),
        ]
    ],
)
def test_data_config_properties(cls: ReprMixin):
    repr_fields = cls._repr_fields # pyright: ignore[reportPrivateUsage]
    for f in cls.__dict__:
        if f.startswith("_"):
            continue
        assert f in repr_fields, f"{f} not in _repr_fields for {cls.__class__.__name__}"