from ..datapoints.recorded import RecordedDataPoint
from .recorded_base import RecordedInputFeatureBase

from ..utils.polars import FrameT, eager_item, frame_columns
from ..scaling.data import ScalerFactory
from .._internal.constants import TIME_CL

import polars as pl


from typing import Any, Callable, Optional, cast

class ModularInputFeature(RecordedInputFeatureBase):
    def __init__(
        self,
        base_datapoint: RecordedDataPoint,
        feature_name: str,
        input_length: str,
        modification_expression: Callable[[FrameT, str, str, list[str]], FrameT],
        uses_future_data: bool = False,
        scaler_factory: Optional[ScalerFactory] = None,
        *,
        debug: bool = False,
    ):
        self._sub_feature_names: Optional[list[str]] = None
        if uses_future_data:
            raise NotImplementedError("ModularInputFeature does not support uses_future_data=True yet. Please set uses_future_data=False.")

        super().__init__(base_datapoint, feature_name, input_length, uses_future_data, scaler_factory, debug=debug)

        self.modification_expression = modification_expression
        if self.uses_future_data:
            raise NotImplementedError("ModularInputFeature does not support uses_future_data=True yet. Please set uses_future_data=False.")

    @property
    def _repr_fields(self) -> dict[str, Any]:
        new_fields: dict[str, Any] = {
            "modification_expression": self.modification_expression,
        }
        return super()._repr_fields | new_fields

    def _generate_sub_feature_names(self) -> Optional[list[str]]:
        return self._sub_feature_names

    def _create_features(self, dataframe: FrameT, grouped: list[str]) -> FrameT:
        features = (
            dataframe
            .rolling(TIME_CL, period=self.input_length.to_string(), closed="both", group_by=grouped)
            .having(pl.col("t0").last().is_not_null())
            .agg(
                pl.col("t0").last(),
                pl.col("time").alias("feature_time"),
                pl.col(self.feature_name)
            )
            .explode(["feature_time", self.feature_name], empty_as_null=True)
            .drop("time")
        )

        features = self.modification_expression(features, "feature_time", self.feature_name, ["t0"] + grouped) # pyright: ignore[reportArgumentType]

        features = (
            features
            .sort(["t0"] + grouped) # somehow the sort is needed to avoid the groupby aggregation to fall back to the in-memory engine
            .drop("feature_time", strict=False)
            .group_by(
                "t0",
                *grouped
            )
            .agg(
                pl.all().explode(empty_as_null=True) # if the modification_expression returns multiple values per t0, we need to stack them into a list
                # also explode in case it is a list of values
            )
        )

        if self._sub_feature_names is None:
            sub_names: list[str] = []
            features_only = features.drop("t0", *grouped, strict=False)
            cols = frame_columns(features_only)
            for c in cols:   
                col_len = eager_item(features_only, pl.col(c).first().explode(empty_as_null=True).len())
                if c == self.feature_name:
                    n = c
                else:
                    n = f"{self.feature_name}_{c}"
                if col_len == 1:
                    sub_names.append(n)
                else:
                    sub_names.extend([f"{n}_[{i}]" for i in range(col_len)])
            self._sub_feature_names = sub_names

        return cast(FrameT, features)