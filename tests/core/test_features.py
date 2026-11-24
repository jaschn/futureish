from typing import Callable, Optional, Sequence

import pytest
import polars as pl
from datetime import datetime, timedelta

from futureish.datapoints import RecordedDataPoint
from futureish.datasets import Dataset
from futureish.features import InputFeatureBase, TargetFeature, RecordedInputFeature, SinCosTimeFeature, ModularInputFeature, BucketInputFeature
from futureish.features.sparse import SparseInputFeature
from futureish.scaling import NoopScaler, ZScoreScaler, DiffScaler, ChainedScaler
from futureish.scaling.base import ScalerBase
from futureish.scaling.data import ScalerFactory

def create_dataset(
      input_features: Optional[Sequence[InputFeatureBase]] = None,
      target_features: Optional[Sequence[TargetFeature]] = None,
      datapoint_name: str = "value_a",
    ):
    dp = RecordedDataPoint(datapoint_name, step_size="1m")
    if input_features is None:
        input_features = [RecordedInputFeature(dp, feature_name="input_feature")]
    if target_features is None:
        target_features = [TargetFeature(dp, feature_name="target_feature")]

    raw_data = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2023, 1, 1, 0, 0, 0),
                end=datetime(2023, 1, 1, 23, 59, 0),
                interval="1m",
                eager=True,
            )
        }
    ).with_columns(
        pl.int_range(0, pl.len()).alias("value_a")
    )

    dataset = Dataset(
        raw_data=raw_data,
        input_features=input_features,
        target_features=target_features,
    )

    return dataset

def dataset_verify(dataset: Dataset):
    train_dataset = dataset.to_train_dataset()
    _, prediction_dataset = dataset.split_training_and_test_data(0.1)

    train_data = train_dataset.create_training_data()
    assert train_dataset.config.sub_features_names is not None, "Sub features names should not be None"
    assert len(train_dataset.config.sub_features_names) == train_data.select(pl.col("X").first().arr.len()).item()
    prediction_dataset.create_all_single_points()
    prediction_dataset.create_all_trajectories(pred_len="10m")
    assert prediction_dataset.config.sub_features_names == train_dataset.config.sub_features_names

input_lengths = ["0m", "1m", "2m", "3m", "4m", "5m", "10m"]
input_lengths_no0 = ["1m", "2m", "3m", "4m", "5m", "10m"]
scaler_factories = [
        None, NoopScaler, ZScoreScaler, DiffScaler, ChainedScaler.from_scalers(NoopScaler, ZScoreScaler, DiffScaler), ChainedScaler.from_scalers(DiffScaler, NoopScaler, ZScoreScaler)
    ]

@pytest.mark.parametrize(
    "input_len",
    input_lengths
)
@pytest.mark.parametrize(
    "uses_future_data",
    [
        True, False
    ]
)
@pytest.mark.parametrize(
    "scaler_factory",
    scaler_factories
)
def test_recorded_input_feature(input_len: str, uses_future_data: bool, scaler_factory: Optional[ScalerFactory]):
    dp = RecordedDataPoint("value_a", step_size="1m", known_future=uses_future_data)
    input_features: list[InputFeatureBase] = [
        RecordedInputFeature(dp, feature_name="input_feature", input_length=input_len, uses_future_data=uses_future_data, scaler_factory=scaler_factory, debug=True),
    ]
    dataset = create_dataset(input_features=input_features)
    dataset_verify(dataset)

@pytest.mark.parametrize(
    "input_len",
    input_lengths_no0
)
@pytest.mark.parametrize(
    "uses_future_data",
    [
        True, False
    ]
)
@pytest.mark.parametrize(
    "scaler_factory",
    scaler_factories
)
@pytest.mark.parametrize(
    "reduction_factor",
    [1, 2, 3, 4, 5, 10]
)
def test_sparse_input_feature(input_len: str, uses_future_data: bool, scaler_factory: Optional[ScalerFactory], reduction_factor: int):
    dp = RecordedDataPoint("value_a", step_size="1m", known_future=uses_future_data)
    input_features: Sequence[InputFeatureBase] = [
        SparseInputFeature(dp, feature_name="input_feature", input_length=input_len, uses_future_data=uses_future_data, scaler_factory=scaler_factory, reduction_factor=reduction_factor),
    ]
    dataset = create_dataset(input_features=input_features)
    dataset_verify(dataset)

def test_sin_cos_feature():
    input_features = [
        SinCosTimeFeature(second=True, minute=True, hour=True, weekday=True, month=True, ordinal_day=True),
    ]
    dataset = create_dataset(input_features=input_features)
    dataset_verify(dataset)


def noop(df: pl.LazyFrame, feature_time: str, feature_name: str, group_cols: list[str]) -> pl.LazyFrame:
    return df

def first_expression(df: pl.LazyFrame, feature_time: str, feature_name: str, group_cols: list[str]) -> pl.LazyFrame:
    df = (
        df
        .group_by(
            *group_cols
        )
        .agg(
            pl.col(feature_name).first().alias("tmp")
        )
    )
    return df

def second_expression(df: pl.LazyFrame, feature_time: str, feature_name: str, group_cols: list[str]) -> pl.LazyFrame:
    df = (
        df
        .group_by(
            *group_cols
        )
        .agg(
            pl.col(feature_name).gather_every(3)
        )
    )
    return df

def third_expression(df: pl.LazyFrame, feature_time: str, feature_name: str, group_cols: list[str]) -> pl.LazyFrame:
    df = (
        df
        .group_by(
            *group_cols
        )
        .agg(
            pl.col(feature_name).gather_every(3).alias("tmp"),
            pl.col(feature_name).min().alias("tmp2")
        )
    )
    return df

def fourth_expression(df: pl.LazyFrame, feature_time: str, feature_name: str, group_cols: list[str]) -> pl.LazyFrame:
    df = (
        df
        .with_columns(
            pl.col(feature_name) ** 2,
            pl.col(feature_name).alias("tmp")
        )
    )
    return df

@pytest.mark.parametrize(
    "input_len",
    input_lengths_no0
)
@pytest.mark.parametrize(
    "scaler_factory",
    scaler_factories
)
@pytest.mark.parametrize(
    "modification_expression",
    [
        noop,
        first_expression,
        second_expression,
        third_expression,
        fourth_expression
    ]
)
def test_modular_input_feature(input_len: str, scaler_factory: Optional[ScalerFactory], modification_expression: Callable[[pl.LazyFrame, str, str, list[str]], pl.LazyFrame]):
    dp = RecordedDataPoint("value_a", step_size="1m")
    input_features = [
        ModularInputFeature(
            base_datapoint=dp,
            feature_name="input_feature",
            input_length=input_len,
            scaler_factory=scaler_factory,
            modification_expression=modification_expression
        ),
    ]
    dataset = create_dataset(input_features=input_features)
    dataset_verify(dataset)

def agg_expr_mean(expr: pl.Expr, feature_time: str) -> pl.Expr:
    return expr.mean()

def agg_expr_noop(expr: pl.Expr, feature_time: str) -> pl.Expr:
    return expr

@pytest.mark.parametrize(
    "input_len",
    input_lengths_no0
)
@pytest.mark.parametrize(
    "scaler_factory",
    scaler_factories
)
@pytest.mark.parametrize(
    "agg_expr",
    [
        agg_expr_mean,
        agg_expr_noop
    ]
)
def test_bucket_input_feature(input_len: str, scaler_factory: Optional[ScalerFactory], agg_expr: Callable[[pl.Expr, str], pl.Expr]):
    dp = RecordedDataPoint("value_a", step_size="1m")
    input_features: list[InputFeatureBase] = [
        BucketInputFeature(
            base_datapoint=dp,
            feature_name="input_feature",
            input_length=input_len,
            scaler_factory=scaler_factory,
            group_by_window="5m",
            agg_expr=agg_expr,
        ),
    ]
    dataset = create_dataset(input_features=input_features)
    dataset_verify(dataset)

@pytest.mark.parametrize(
        "window_length",
        [1, 2, 3, 4, 5, 6, 8]
)
@pytest.mark.parametrize(
    "input_length",
    [2, 3, 4, 5, 6]
)
def test_bucket_input_feature_impl(window_length: int, input_length: int):
    import numpy as np
    dp = RecordedDataPoint("value_a", step_size="1m")
    input_features: list[InputFeatureBase] = [
        BucketInputFeature(
            base_datapoint=dp,
            feature_name="input_feature",
            input_length=f"{input_length - 1}m", # 9m history + t0 = 10 datapoints
            group_by_window=f"{window_length}m",
            agg_expr=lambda expr, feature_time: expr.len(),
        ),
    ]
    xf_shape = input_length // window_length
    expected_lens = [window_length] * (input_length // window_length)
    if input_length % window_length > 0:
        xf_shape += 1
        expected_lens += [input_length % window_length]
    expected_lens = np.array(expected_lens)
    dataset = create_dataset(input_features=input_features)
    dataset_verify(dataset)
    X = dataset.to_train_dataset().create_training_data().get_column("X").to_numpy()
    assert X.shape[1] == xf_shape, f"Expected {xf_shape} features, found {X.shape[1]}"
    assert (X == expected_lens).all()

@pytest.mark.parametrize(
        "window_length",
        [1, 2, 3, 4, 5, 6, 8]
)
@pytest.mark.parametrize(
    "input_length",
    [2, 3, 4, 5, 6]
)
def test_bucket_input_feature_impl2(window_length: int, input_length: int):
    import numpy as np
    dp = RecordedDataPoint("value_a", step_size="1m")
    input_features: list[InputFeatureBase] = [
        BucketInputFeature(
            base_datapoint=dp,
            feature_name="input_feature",
            input_length=f"{input_length - 1}m", # 9m history + t0 = 10 datapoints
            group_by_window=f"{window_length}m",
            agg_expr=lambda expr, feature_time: expr.max(),
        ),
    ]
    dataset = create_dataset(input_features=input_features)
    dataset_verify(dataset)
    X = dataset.to_train_dataset().create_training_data().get_column("X").to_numpy()
    for i in range(X.shape[0]):
        XX = X[i]
        for ii in range(XX.shape[0]-1):
            assert XX[ii] > XX[ii+1], f"Expected all values to be equal, but found {XX[ii]} <= {XX[ii+1]}"
