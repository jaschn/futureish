import pytest
import polars as pl
from datetime import datetime
from typing import Callable, Optional, cast

from futureish.datapoints.recorded import RecordedDataPoint
from futureish.datasets.dataset import Dataset
from futureish.features.bucket import BucketInputFeature
from futureish.features.businessday import BusinessDayFeature
from futureish.features.modular import ModularInputFeature
from futureish.features.recorded import RecordedInputFeature
from futureish.features.sincos_time import SinCosTimeFeature
from futureish.features.sparse import SparseInputFeature
from futureish.features.target import TargetFeature
from futureish.scaling.chained import ChainedScaler
from futureish.scaling.difference import DiffScaler
from futureish.scaling.noop import NoopScaler
from futureish.scaling.z_score import ZScoreScaler

def generate_raw_data(interval: str, time_zone: Optional[str] = None) -> pl.DataFrame:
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    data = (
        pl.DataFrame()
        .select(
                pl.lit(start_time).cast(pl.Datetime(time_unit="ns" if interval.endswith("s") else "ms")).dt.truncate(interval).alias("time").alias("start")  
            )
        .with_columns(
            (
                pl.col("start") + 
                (pl.col("start").dt.offset_by(interval) - pl.col("start")) * 1100
            ).alias("end")
        )
    )
    data = (
        data
        .select(
            time=pl.datetime_range(
                start=pl.col("start").first(),
                end=pl.col("end").first(),
                interval=interval,
                time_zone=time_zone
            )
        )
        .with_columns(
            pl.repeat(1, pl.len()).alias("value"),
            pl.repeat(2, pl.len()).alias("target"),
        )
        .head(1000)
    )
    
    assert data.height == 1000, f"Data height is {data.height}, expected 1000"
    return data

def dataset_verify(dataset: Dataset, input_len: int, unit: list[str]):
    train_dataset = dataset.to_train_dataset()
    _, prediction_dataset = dataset.split_training_and_test_data(0.1)

    train_data = train_dataset.create_training_data()
    assert train_data.height > 0, f"Training data height is {train_data.height}, expected > 0"
    assert train_data.select(pl.col("X").first().arr.len()).item() > 0, f'Training data features length is {train_data.select(pl.col("X").first().arr.len()).item()}, expected > 0'
    assert train_dataset.config.sub_features_names is not None, "Sub features names should not be None"
    assert len(train_dataset.config.sub_features_names) == train_data.select(pl.col("X").first().arr.len()).item()
    prediction_dataset.create_all_single_points()
    pred_len = "".join([f"{10*input_len}{u}" for u in unit])
    traj = prediction_dataset.create_all_trajectories(pred_len=pred_len)
    assert traj.height > 0, f"Trajectory height is {traj.height}, expected > 0"
    assert prediction_dataset.config.sub_features_names == train_dataset.config.sub_features_names

scaler_factories = [
        None, NoopScaler, ZScoreScaler, DiffScaler, ChainedScaler.from_scalers(NoopScaler, ZScoreScaler, DiffScaler), ChainedScaler.from_scalers(DiffScaler, NoopScaler, ZScoreScaler)
    ]

base_time_units = ["ns", "us", "ms", "s", "m", "h", "d", "w", "mo", "q", "y"]
possible_base_time_units_combination = [
    [unit] for unit in base_time_units
] + [
    [base_time_units[i-1], base_time_units[i]]
    for i in range(1, len(base_time_units))
    if base_time_units[i] not in ["w", "mo", "q", "y"] and base_time_units[i-1] not in ["w", "mo", "q", "y"]
]

@pytest.mark.parametrize("input_len", [1, 2, 3], ids=lambda x: f"input_len={x}")
@pytest.mark.parametrize("interval", possible_base_time_units_combination, ids=lambda x: f"interval={x}")
@pytest.mark.parametrize("feature_input_len", [1, 2, 3], ids=lambda x: f"feature_input_len={x}")
@pytest.mark.parametrize("scaler_factory", scaler_factories, ids=lambda x: f"scaler_factory={x.__name__ if x is not None else 'None'}")
@pytest.mark.parametrize("time_zone", [None, "UTC"], ids=lambda x: f"time_zone={x}")
def test_dataset_generation(input_len: int, interval: list[str], scaler_factory, feature_input_len: int, time_zone: str):
    step_size = "".join([f"{input_len}{unit}" for unit in interval])
    raw_data = generate_raw_data(step_size, time_zone=time_zone)
    assert raw_data.height == 1000
    dtype = cast(pl.Datetime, raw_data.collect_schema().get("time"))
    assert dtype.time_zone == time_zone

    dp_value = RecordedDataPoint(name="value", step_size=step_size, known_future=True)
    dp_target = RecordedDataPoint(name="target", step_size=step_size)
    input_len_feature = "".join([f"{feature_input_len*input_len}{unit}" for unit in interval])
    group_by_window = f"{input_len}{interval[-1]}"
    input_features = [
        RecordedInputFeature(base_datapoint=dp_value, feature_name="value_recorded", input_length=input_len_feature, scaler_factory=scaler_factory),
        SinCosTimeFeature(second=True, minute=True, hour=True, weekday=True, ordinal_day=True, month=True),
        BusinessDayFeature(day_history=7),
        SparseInputFeature(base_datapoint=dp_value, feature_name="value_sparse", input_length=input_len_feature, scaler_factory=scaler_factory),
        ModularInputFeature(base_datapoint=dp_value, feature_name="value_modular", input_length=input_len_feature, modification_expression=lambda df, time_col, feature_col, group_cols: df.with_columns((pl.col(feature_col) * 2)), scaler_factory=scaler_factory),
        BucketInputFeature(base_datapoint=dp_value, feature_name="value_bucket", input_length=input_len_feature, group_by_window=group_by_window, agg_expr=lambda expr, feature_time: expr ** 2, scaler_factory=scaler_factory)
    ]
    target_features = [
        TargetFeature(base_datapoint=dp_target, feature_name="target_feature", scaler_factory=scaler_factory)
    ]
    dataset = Dataset(
        raw_data=raw_data,
        input_features=input_features,
        target_features=target_features,
        debug=True
    )
    dataset_verify(dataset, input_len, interval)