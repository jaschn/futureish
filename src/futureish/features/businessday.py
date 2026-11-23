from typing import Any

from ..datapoints.time import TimeDataPoint
from .input_base import InputFeatureBase
from ..utils.polars import FrameT
import polars as pl

class BusinessDayFeature(InputFeatureBase):
    def __init__(
                self,
                day_history: int = 0,
            ):
        super().__init__(
            base_datapoint=TimeDataPoint(),
            feature_name="BusinessDay_feature",
        )
        if day_history < 0:
            raise ValueError("day_history must be non-negative")
        self.day_history = day_history

    @property
    def _repr_fields(self) -> dict[str, Any]:
        new_fields: dict[str, Any] = {
            "day_history": self.day_history,
        }
        return super()._repr_fields | new_fields

    def _generate_sub_feature_names(self) -> list[str]:
        sub_features: list[str] = []
        for i in range(self.day_history + 1):
            sub_features.append(f"{self.feature_name}_day_{i}")
        return sub_features

    def _supports_sparse_input(self) -> bool:
        return True

    def _create_features(self, dataframe: FrameT, grouped: list[str]) -> FrameT:      
        df = (
            dataframe
            .filter(
                pl.col("t0").is_not_null()
            )
            .select(
                "t0",
                *grouped
            )
            .with_columns(
                *[
                    pl.col("t0").dt.offset_by(f"{i}d").dt.is_business_day().alias(f"{self.feature_name}_day_{i}")
                    for i in range(self.day_history + 1)
                ]
            )
        )

        return df
            