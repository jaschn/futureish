from ..datapoints.time import TimeDataPoint
from ..utils.polars import FrameT, eager_item, add_cos_sin_features
from .input_base import InputFeatureBase


import polars as pl


from typing import Any, Optional

class SinCosTimeFeature(InputFeatureBase):
    def __init__(
                self,
                *,
                second: bool = False,
                minute: bool = False,
                hour: bool = False,
                weekday: bool = False,
                ordinal_day: bool = False,
                month: bool = False,
            ):
        dp = TimeDataPoint()
        super().__init__(
            base_datapoint=dp,
            feature_name="SinCos_time_feature",
        )
        if not any([second, minute, hour, weekday, ordinal_day, month]):
            raise ValueError("At least one of second, minute, hour, weekday, ordinal_day, or month must be True")
        
        self.second = second
        self.minute = minute
        self.hour = hour
        self.weekday = weekday
        self.month = month
        self.ordinal_day = ordinal_day


    @property
    def _repr_fields(self) -> dict[str, Any]:
        new_fields: dict[str, Any] = {
            "second": self.second,
            "minute": self.minute,
            "hour": self.hour,
            "weekday": self.weekday,
            "month": self.month,
            "ordinal_day": self.ordinal_day,
        }
        return super()._repr_fields | new_fields

    def _generate_sub_feature_names(self) -> list[str]:
        sub_features: list[str] = []
        if self.second:
            sub_features.extend(["second_sin", "second_cos"])
        if self.minute:
            sub_features.extend(["minute_sin", "minute_cos"])
        if self.hour:
            sub_features.extend(["hour_sin", "hour_cos"])
        if self.weekday:
            sub_features.extend(["weekday_sin", "weekday_cos"])
        if self.month:
            sub_features.extend(["month_sin", "month_cos"])
        if self.ordinal_day:
            sub_features.extend(["ordinal_day_sin", "ordinal_day_cos"])
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
            .pipe(
                add_cos_sin_features,
                time_column="t0",
                second=self.second,
                minute=self.minute,
                hour=self.hour,
                weekday=self.weekday,
                month=self.month,
                ordinal_day=self.ordinal_day,
                year=False,
            )
        )

        return df
            