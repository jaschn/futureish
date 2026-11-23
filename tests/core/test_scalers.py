#type: ignore

from pathlib import Path

import pytest
import polars as pl

from futureish.scaling import ChainedScaler, DiffScaler, NoopScaler, ZScoreScaler
from futureish.scaling.data import ScalerFactory

@pytest.fixture
def taxi_data() -> pl.DataFrame:
    base = Path(__file__).parent.parent / "test_data"
    taxi_data_file = base / "taxi_data.parquet"
    if not taxi_data_file.exists():
        base.mkdir(parents=True, exist_ok=True)
        pl.read_csv("https://raw.githubusercontent.com/unit8co/darts/refs/heads/master/datasets/taxi_new_york_passengers.csv", try_parse_dates=True).rename({"#Passengers": "value"}).write_parquet(taxi_data_file)
    return pl.read_parquet(Path(__file__).parent.parent / "test_data" / "taxi_data.parquet")

@pytest.mark.parametrize(
        "scaler_class",
            [
                DiffScaler,
                ZScoreScaler,
                ChainedScaler.from_scalers(DiffScaler, ZScoreScaler),
                ChainedScaler.from_scalers(ZScoreScaler, DiffScaler),
                NoopScaler,
            ]
        )
def test_scaling(scaler_class: ScalerFactory, taxi_data: pl.DataFrame):
    scaler = scaler_class(taxi_data, "value", "target_value")

    scaled = scaler.scale(taxi_data)

    scaled_fixed = (
        scaled
        .join(
            taxi_data,
            on="time",
            how="right"
        )
        .select(
            "time",
            pl.when(pl.col("time") != pl.col("time").min()).then("target_value").otherwise("value"),
            pl.when(pl.col("time") == pl.col("time").min()).then(pl.col("time")).otherwise(pl.lit(None)).alias("t0")
        )
    )

    inv_scaled = scaler.inverse_scale(scaled_fixed).drop("t0").sort("time")

    for r, r_orig in zip(inv_scaled.rows(), taxi_data.rows(), strict=True):
        assert r[0] == r_orig[0], f"Time mismatch: {r[0]} != {r_orig[0]}"
        assert r[1] == pytest.approx(r_orig[1], rel=1e-10), f"Value mismatch: {r[1]} != {r_orig[1]}"


def test_chained_scaler_requires_at_least_one_scaler(taxi_data: pl.DataFrame):

    with pytest.raises(AssertionError, match="ChainedScaler requires at least one scaler"):
        ChainedScaler(taxi_data, "value", "target_value")


def test_chained_scaler_inverse_scale_raw_column_already_present(taxi_data: pl.DataFrame):
    """Regression test: InferenceModel builds one working frame per
    recursive step that can carry a target's raw column (base_datapoint.name)
    alongside its scaled feature column at the same time - this happens
    whenever an InputFeature reads the very same base_datapoint a
    TargetFeature predicts (the canonical autoregressive setup, e.g. a
    process variable that's also modeled as its own target).

    With ZScoreScaler first in the chain, inverse_scale() takes the
    t0-anchor-rescale path (_rescale_t0_anchor/_restore_t0_anchor), which
    used to plainly `.rename()` the feature column onto the raw column's
    name - raising DuplicateError whenever that raw name was already present,
    exactly as it is here. See ChainedScaler._safe_rename().
    """

    chain = ChainedScaler(taxi_data, "value", "target_value", ZScoreScaler, DiffScaler)
    scaled = chain.scale(taxi_data)

    # Simulate InferenceModel's working frame: the scaled "target_value"
    # column coexists with a raw "value" passthrough column, as it would for
    # an InputFeature sharing this base_datapoint.
    scaled_with_raw_column = (
        scaled
        .join(taxi_data.select("time", "value"), on="time", how="left")
        .with_columns(
            pl.when(pl.col("time") == pl.col("time").min()).then(pl.col("time")).otherwise(pl.lit(None)).alias("t0")
        )
    )

    result = chain.inverse_scale(scaled_with_raw_column)

    assert "value" in result.columns
    assert "target_value" not in result.columns


def test_chained_scaler_inverse_scale_batches_multiple_t0_groups(taxi_data: pl.DataFrame):
    """Regression test: ChainedScaler.inverse_scale()'s t0-anchor-rescale
    path (taken whenever a ZScoreScaler runs before a later stage, e.g.
    ZScoreScaler then DiffScaler) used to locate its anchor row via a
    frame-wide pl.col("t0").max() - correct for a single global t0, but
    picking the wrong (highest) anchor for every other t0 once several
    different t0's are inverse-scaled together in one call, grouped via
    `over` (see InferenceModel.predict(), which does exactly this to unscale
    many one-step predictions at once).

    Checks that inverse-scaling several t0 groups together in one call, via
    `over`, gives the same per-group result as inverse-scaling each group
    alone, one call at a time.
    """

    chain = ChainedScaler(taxi_data, "value", "target_value", ZScoreScaler, DiffScaler)
    scaled = chain.scale(taxi_data)

    # Build one isolated [anchor row, one-step-ahead prediction row] pair per
    # chosen t0 - "t0" itself stays null except on the anchor row (the
    # convention every scaler assumes), grouped instead via a separate
    # "group" id shared by both rows of a pair - exactly how
    # InferenceModel._unscale_one_step batches its frame.
    t0_indices = [3, 7, 11]
    groups = [
        pl.DataFrame({
            "time": [taxi_data[i, "time"], scaled[i + 1, "time"]],
            "t0": [taxi_data[i, "time"], None],
            "group": [i, i],
            "target_value": [float(taxi_data[i, "value"]), scaled[i + 1, "target_value"]],
        })
        for i in t0_indices
    ]

    batched_result = chain.inverse_scale(pl.concat(groups), over="group")

    for group, i in zip(groups, t0_indices):
        solo_result = chain.inverse_scale(group.drop("group"))
        expected = solo_result.filter(pl.col("t0").is_null()).select("time", "value")
        actual = batched_result.filter(pl.col("group") == i, pl.col("t0").is_null()).select("time", "value")
        assert actual.equals(expected), f"batched result for group={i} diverged from the solo (single-group) result"


def test_chained_scaler_zscore_not_at_start_raises(taxi_data: pl.DataFrame):
    # ZScoreScaler appearing in the middle (not first, not last) of the chain is
    # explicitly unsupported by the inverse_scale "scale_fix" logic.
    chain = ChainedScaler(taxi_data, "value", "target_value", NoopScaler, ZScoreScaler, DiffScaler)
    scaled = chain.scale(taxi_data)
    scaled = scaled.with_columns(
        pl.when(pl.col("time") == pl.col("time").min()).then(pl.col("time")).otherwise(pl.lit(None)).alias("t0")
    )

    with pytest.raises(Exception, match="ZScoreScaler not at the beginning of the chain"):
        chain.inverse_scale(scaled)