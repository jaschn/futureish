import warnings

from futureish.utils.timedeltaoperator import TimeDeltaOperator

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
        self.group_by_window = TimeDeltaOperator(group_by_window)
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

    def _bucket_modification_expression(self, x: FrameT, feature_time: str, feature: str, grouped: list[str]) -> FrameT:
        bounds = self.group_by_window.bounds_up_to(self.input_length)

        def get_mask(i: int):
            return (
                pl.col(feature_time).is_between(
                    bounds[i + 1].offset_column_backward(pl.col("t0")),
                    bounds[i].offset_column_backward(pl.col("t0")),
                    closed="right"
                ).cast(pl.Int8)
        )

        x1 = (
            x
            .with_columns(
                pl.sum_horizontal([  
                    (
                        pl.lit(i) * get_mask(i) # mask is 1 if the feature_time is in the bucket i, otherwise 0
                    )
                    for i in range(len(bounds) - 1)
                ]).alias("window_index"),
            )
        )

        x2 = (
            x1
            .sort(feature_time)
            .group_by(grouped + ["window_index"])
            .agg(
                pl.col(feature)
                .pipe(self.agg_expr, feature_time).alias(feature)
            )
        )

        if self._sub_feature_names is None:
            if self.agg_expr(pl.col(feature), feature_time).meta.is_scalar():
                self._sub_feature_names = [w.to_string() for w in bounds[1:]]
            else:
                dtype = x2.collect_schema().get(feature)
                if isinstance(dtype, pl.List):
                    lens = []
                    tmp = x2.filter(pl.col("t0") == pl.col("t0").first()).select("window_index", pl.col(feature).list.len()).sort("window_index")
                    tmp = to_eager(tmp)
                    for r in tmp.iter_rows():
                        lens.extend(
                                [
                                    "bucket_[-" + bounds[r[0] + 1].to_string() + "]" + f"_[{i}]" for i in range(r[1])
                                ]
                            )
                    self._sub_feature_names = lens
                else:
                    warnings.warn("BucketInputFeature: agg_expr does not return a list, but the feature is not scalar. the sub_feature_names will not be generated correctly.")
            
        buckets_str = {str(i): "bucket_[-" + w.to_string() + "]" for i, w in enumerate(bounds[1:])}

        x3 = (
            x2
            .pivot(
                on="window_index",
                on_columns=list(range(len(bounds) - 1)),
                index=grouped,
                values=feature
            )
            .rename(buckets_str)
            .select(
                *grouped,
                *list(buckets_str.values()) # make sure the column order matches the sub_feature_names order
            )
        )

        return x3