import warnings

from ..datapoints.recorded import RecordedDataPoint
from .modular import ModularInputFeature

from ..utils.polars import FrameT, to_eager
from ..scaling.data import ScalerFactory


import polars as pl


from typing import Any, Callable, Optional

class BucketInputFeature(ModularInputFeature):
    def __init__(
        self,
        base_datapoint: RecordedDataPoint,
        feature_name: str,
        input_length: str,
        group_by_window: str,
        agg_expr: Callable[[pl.Expr, str], pl.Expr], # expression to aggregate, feature_time for reference. Also t0 available in the expression
        uses_future_data: bool = False,
        scaler_factory: Optional[ScalerFactory] = None,
        *,
        debug: bool = False,
    ):
        super().__init__(
            base_datapoint=base_datapoint,
            feature_name=feature_name,
            input_length=input_length,
            uses_future_data=uses_future_data,
            modification_expression=self._bucket_modification_expression,
            scaler_factory=scaler_factory,
            debug=debug,
        )
        self.group_by_window = group_by_window
        self.agg_expr = agg_expr
        self._buckets_cache: Optional[dict[int, str]] = None
        if self.uses_future_data:
            raise NotImplementedError("BucketInputFeature does not support uses_future_data=True yet. Please set uses_future_data=False.")

    @property
    def _repr_fields(self) -> dict[str, Any]:
        new_fields: dict[str, Any] = {
            "group_by_window": self.group_by_window,
            "agg_expr": self.agg_expr,
        }
        old_fields = super()._repr_fields
        old_fields["modification_expression"] = "BucketInputFeature internal"  # Remove modification_expression from the old fields
        return old_fields | new_fields

    def _generate_split_point(self, expr: pl.Expr, n_offsets: int) -> pl.Expr:
        offset = pl.lit("-" + self.group_by_window)
        for _ in range(n_offsets):
            expr = expr.dt.offset_by(offset)
        return expr

    def _generate_n_buckets(self) -> dict[int, str]:
        if self._buckets_cache is not None:
            return self._buckets_cache

        n_splits = 0
        covered = False
        while not covered:
            n_splits += 1
            covered = (
                pl.DataFrame()
                .select(
                    pl.lit(0).cast(self.time_column_dtype).alias("feature_time"),
                )
                .select(
                    pl.col("feature_time").pipe(self._generate_split_point, n_splits),
                    pl.lit(0).cast(self.time_column_dtype).dt.offset_by("-" + self.input_length.to_string()).alias("input_window_start"),
                )
                .select(
                    pl.col("feature_time") < pl.col("input_window_start")
                )
                .item()
            )

        buckets = {i: f"{self.feature_name}_bucket_{i}" for i in range(n_splits)}

        self._buckets_cache = buckets
        return buckets

    def _bucket_modification_expression(self, x: FrameT, feature_time: str, feature: str, grouped: list[str]) -> FrameT:
        buckets = self._generate_n_buckets()

        def get_mask(i: int):
            return (
                pl.col(feature_time).is_between(
                    pl.col("t0").pipe(self._generate_split_point, i + 1),
                    pl.col("t0").pipe(self._generate_split_point, i),
                    closed="right"
                )
        )

        x1 = (
            x
            .with_columns(
                pl.max_horizontal([  
                    (
                        pl.lit(0) + (pl.lit(i) + 1) * get_mask(i).cast(pl.Int8) - 1
                    )
                    for i in buckets.keys()
                ]).alias("window_index"),
            )
            .with_columns(
                pl.when(pl.col("window_index") >= 0).then(pl.col("window_index"))
            )
        )

        x2 = (
            x1
            .group_by(grouped + ["window_index"])
            .agg(
                pl.col(feature)
                .pipe(self.agg_expr, feature_time).alias(feature)
            )
        )

        if self._sub_feature_names is None:
            if self.agg_expr(pl.col(feature), feature_time).meta.is_scalar():
                self._sub_feature_names = list(buckets.values())
            else:
                dtype = x2.collect_schema().get(feature)
                if isinstance(dtype, pl.List):
                    lens = []
                    tmp = x2.filter(pl.col("t0") == pl.col("t0").first()).select("window_index", pl.col(feature).list.len()).sort("window_index")
                    tmp = to_eager(tmp)
                    for r in tmp.iter_rows():
                        lens.append(buckets[r[0]] + f"_[{i}]" for i in range(r[1]))
                else:
                    warnings.warn("BucketInputFeature: agg_expr does not return a list, but the feature is not scalar. the sub_feature_names will not be generated correctly.")
            
        buckets_str = {str(k): v for k, v in buckets.items()}

        x3 = (
            x2
            .pivot(
                on="window_index",
                on_columns=list(buckets.keys()),
                index=grouped,
                values=feature
            )
            .rename(buckets_str)
            .select(
                *grouped,
                *list(buckets.values()) # make sure the column order matches the sub_feature_names order
            )
        )

        return x3