from futureish.utils.time_utils import parse_timedelta
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from futureish.utils.timedelta import TimeDelta

import polars as pl

from datetime import timedelta
from functools import total_ordering
from typing import Optional, Union

@total_ordering
class TimeDeltaOperator:
    def __init__(self, delta: Optional[Union[str, timedelta, "TimeDelta", "TimeDeltaOperator"]] = None):
        self.orig_delta = delta
        self.td = parse_timedelta(delta)

    def to_string(self) -> str:
        return self.td.to_string()

    def __eq__(self, other: object) -> bool:
        return self.td == other

    def __lt__(self, other: object) -> bool:
        return self.td < other

    def __gt__(self, other: object) -> bool:
        return self.td > other

    def __str__(self) -> str:
        return self.td.to_string()

    def __repr__(self) -> str:
        return self.td.to_string()

    def __hash__(self) -> int:
        return hash(self.td)

    def __add__(self, other: "TimeDeltaOperator") -> "TimeDeltaOperator":
        return TimeDeltaOperator(self.td + other.td)

    def is_negative(self) -> bool:
        return self.td.negative

    def negated(self) -> "TimeDeltaOperator":
        return TimeDeltaOperator(self.td.negated())

    def positive(self) -> "TimeDeltaOperator":
        return TimeDeltaOperator(self.td.positive())

    def truncate_column(self, time_cl: pl.Expr) -> pl.Expr:
        return time_cl.dt.truncate(self.td.to_string())

    def offset_column_forward(self, time_cl: pl.Expr) -> pl.Expr:
        return time_cl.dt.offset_by(self.td.to_string())

    def offset_column_backward(self, time_cl: pl.Expr) -> pl.Expr:
        return time_cl.dt.offset_by("-" + self.td.to_string())

    def is_zero(self) -> bool:
        return self.td.is_zero()

    def is_multiple_of(self, other: "TimeDeltaOperator") -> bool:
        return self.td.is_multiple_of(other.td)

    def bounds_up_to(self: "TimeDeltaOperator", end: "TimeDeltaOperator") -> list["TimeDeltaOperator"]:
        current_pos = self
        range_td = [TimeDeltaOperator()]
        while current_pos <= end:
            range_td.append(current_pos)
            current_pos = current_pos + self
        range_td.append(current_pos)
        return range_td

    @staticmethod
    def range(start: "TimeDeltaOperator", end: "TimeDeltaOperator", step: "TimeDeltaOperator") -> list["TimeDeltaOperator"]:
        if step.is_zero():
            raise ValueError("Step cannot be zero")
        if step.is_negative():
            raise ValueError("Step cannot be negative")
        if start > end:
            raise ValueError(f"Cannot create range from {start} to {end} because {start} is greater than {end}")

        current_pos = start
        range_td = []
        while current_pos <= end:
            range_td.append(current_pos)
            current_pos = current_pos + step
        return range_td