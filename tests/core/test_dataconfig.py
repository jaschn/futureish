#type: ignore
import re

import polars as pl
from datetime import datetime, timedelta
from pytest import fixture
import pytest

from futureish.features.target import TargetFeature
from futureish.features.recorded import RecordedInputFeature
from futureish.datapoints import RecordedDataPoint
from futureish.datasets.config import DataConfig

from polars.testing import assert_frame_equal, assert_series_equal

@fixture
def data_config():
    bd_a = RecordedDataPoint("bd_a", step_size="1m", known_future=True)
    input_features = [
        RecordedInputFeature(bd_a, "feature_1", input_length="1m", uses_future_data=True),
    ]
    target_features = [
        TargetFeature(bd_a, "target_1"),
    ]

    return DataConfig(
        input_features=input_features,
        target_features=target_features,
        time_column_dtype=pl.Datetime(time_unit="us"),
    )

@fixture
def corresponding_data():
    return pl.DataFrame(
        {
            "time": [
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 1, 1, 0, 1),
                datetime(2023, 1, 1, 0, 2),
                datetime(2023, 1, 1, 0, 3),
                datetime(2023, 1, 1, 0, 4),
            ],
            "bd_a": [1, 2, 3, 4, 5],
        }
    )

def test_data_config_init():
    bd_a = RecordedDataPoint("bd_a", step_size="1m", known_future=True)

    input_features = [
        RecordedInputFeature(bd_a, "feature_1", input_length="1m", uses_future_data=True),
    ]
    target_features = [
        TargetFeature(bd_a, "target_1"),
    ]

    with pytest.raises(ValueError, match="time_column_dtype must be a pl.Datetime type"):
        DataConfig(
            input_features=input_features,
            target_features=target_features,
            time_column_dtype=None, # pyright: ignore[reportArgumentType]
        )

    with pytest.raises(ValueError, match="No input features defined"):
        DataConfig(
            input_features=[],
            target_features=target_features,
            time_column_dtype=pl.Datetime(time_unit="ns"),
        )

    with pytest.raises(ValueError, match="No target features defined"):
        DataConfig(
            input_features=input_features,
            target_features=[],
            time_column_dtype=pl.Datetime(time_unit="ns"),
        )

    with pytest.raises(ValueError, match="Feature names must be unique"):
        DataConfig(
            input_features=input_features * 2,
            target_features=target_features,
            time_column_dtype=pl.Datetime(time_unit="ns"),
        )

    bd_a = RecordedDataPoint("bd_a", step_size="1m", known_future=True)
    bd_b = RecordedDataPoint("bd_b", step_size="2m", known_future=True)

    target_features_test = [
        TargetFeature(bd_a, "target_1"),
        TargetFeature(bd_b, "target_2"),
    ]

    with pytest.raises(ValueError, match="All target features must have the same step size"):
        DataConfig(
            input_features=input_features,
            target_features=target_features_test,
            time_column_dtype=pl.Datetime(time_unit="ns"),
        )

def test_data_config_properties(data_config: DataConfig):
    repr_fields = data_config._repr_fields # pyright: ignore[reportPrivateUsage]
    for f in data_config.__dict__:
        if f.startswith("_"):
            continue
        assert f in repr_fields, f"{f} not in _repr_fields"

def test_data_config_fit(data_config: DataConfig, corresponding_data: pl.DataFrame):
    fitted = data_config.fitted(corresponding_data)
    assert fitted.is_fitted, "DataConfig should be marked as fitted after calling fitted()"
    with pytest.raises(ValueError, match="DataConfig is already fitted"):
        fitted.fitted(corresponding_data)
    with pytest.raises(ValueError, match="Base datapoint .* not found in raw data"):
        data_config.fitted(corresponding_data.drop("bd_a"))

def test_data_config_create_data(data_config: DataConfig, corresponding_data: pl.DataFrame):
    fitted = data_config.fitted(corresponding_data)
    corresponding_data_with_t0 = corresponding_data.with_columns(
        pl.when(pl.col("time") == datetime(2023, 1, 1, 0, 2)).then(pl.col("time")).alias("t0")
    )
    corresponding_data_with_t0_grp = pl.concat(
        [
            corresponding_data_with_t0.with_columns(grp=pl.lit(0)),
            corresponding_data_with_t0.with_columns(grp=pl.lit(1)),
        ]
    )

    X = fitted.create_input(corresponding_data_with_t0)
    X_grp = fitted.create_input(corresponding_data_with_t0_grp, grouped="grp")
    assert "grp" in X_grp.columns, "grp not in X_grp columns"
    assert X_grp.unique(subset=pl.exclude("grp")).shape[0] == 1, "X_grp should have the same unique values for each group"
    with pytest.raises(ValueError, match="Dataframe contains duplicate time values but no group column specified"):
        X_dup = fitted.create_input(corresponding_data_with_t0_grp)
    with pytest.raises(ValueError, match="Not enough historical data available before t0 for the configured input length"):
        fitted.create_input(corresponding_data_with_t0.with_columns(pl.col("time").alias("t0")))

    Y = fitted.create_training_target(corresponding_data_with_t0)
    XY = fitted.create_XY(corresponding_data_with_t0)

    assert "X" in XY.columns, "X not in XY columns"
    assert XY.get_column("X").dtype == pl.Array, "X is not of type pl.Array(pl.Float64)"
    assert_series_equal(XY.get_column("X"), X.get_column("X"))
    for f in data_config.target_features:
        assert f.feature_name in XY.columns, f"{f.feature_name} not in XY columns"
        assert XY.get_column(f.feature_name).dtype == pl.Array, f"{f.feature_name} is not of type pl.Array(pl.Float64)"
        assert XY.get_column(f.feature_name).arr.len().max() == 1, f"{f.feature_name} array length is not 1"
        assert_series_equal(XY.get_column(f.feature_name), Y.get_column(f.feature_name))

    with pytest.raises(ValueError, match="No data points found for training. Check your input data and step size."):
        fitted.create_XY(corresponding_data_with_t0.with_columns(pl.lit(None).alias("t0").cast(pl.dtype_of("t0"))))

    with pytest.raises(ValueError, match="No data points found for training. Check your input data and step size."):
        fitted.create_XY(corresponding_data_with_t0.with_columns(pl.lit(None).alias("bd_a").cast(pl.dtype_of("bd_a"))))

    with pytest.raises(ValueError, match="Dataframe must contain 't0' column for create_XY()"):
        XY = fitted.create_XY(corresponding_data)

def test_data_config_sub_feature_names(data_config: DataConfig):
    sub_feature_names = data_config.sub_features_names
    assert sub_feature_names is not None, "sub_features_names should not be None"
    assert len(sub_feature_names) == 3, "sub_features_names should contain 3 features"


def test_input_features_future():
    import numpy as np
    from futureish.datapoints import RecordedDataPoint
    from futureish.features import RecordedInputFeature, TargetFeature
    from futureish.datasets import OnlineDataset, DataConfig
    import polars as pl
    from datetime import datetime, timedelta

    data = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 10),
            interval="1d",
            eager=True,
        )
    }).with_columns(
        pl.int_range(0, pl.len()).alias("feature"),
        pl.int_range(0, pl.len()).alias("value"),
    )
    datapoint_value = RecordedDataPoint(name="value", step_size="1d", known_future=True)
    recorded_target_feature = TargetFeature(datapoint_value, feature_name="value_target")


    datapoint_feature_unknown_future = RecordedDataPoint(name="feature", step_size="1d")
    datapoint_feature_known_future = RecordedDataPoint(name="feature", step_size="1d", known_future=True)
    recorded_input_feature_known_future_no_future = RecordedInputFeature(datapoint_feature_known_future, feature_name="feature_feature", uses_future_data=False)
    recorded_input_feature_known_future_with_future = RecordedInputFeature(datapoint_feature_known_future, feature_name="feature_feature", uses_future_data=True)

    time_dtype = data.collect_schema()["time"]

    with pytest.raises(ValueError, match="Input feature that uses future data must have a base datapoint with known future"):
        recorded_input_feature_unknown_future_with_future = RecordedInputFeature(datapoint_feature_unknown_future, feature_name="feature_feature", uses_future_data=True)
    with pytest.raises(ValueError, match="Input feature feature requires future data to be available"):
        recorded_input_feature_unknown_future_with_future = RecordedInputFeature(datapoint_feature_unknown_future, feature_name="feature_feature")
        recorded_input_feature_unknown_future_with_future.uses_future_data = True
        dataset = DataConfig(input_features=[recorded_input_feature_unknown_future_with_future], target_features=[recorded_target_feature], time_column_dtype=time_dtype)

    dataset = DataConfig(input_features=[recorded_input_feature_known_future_no_future], target_features=[recorded_target_feature], time_column_dtype=time_dtype)
    dataset = DataConfig(input_features=[recorded_input_feature_known_future_with_future], target_features=[recorded_target_feature], time_column_dtype=time_dtype)