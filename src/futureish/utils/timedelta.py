from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Union

import polars as pl
import re

from .._internal.constants import TIME_CL

@dataclass(frozen=True)
class TimeDelta:
    ns: int = 0
    us: int = 0
    ms: int = 0
    s: int = 0
    m: int = 0
    h: int = 0
    d: int = 0
    w: int = 0
    mo: int = 0
    q: int = 0
    y: int = 0
    negative: bool = False

    def to_string(self) -> str:
        parts = []
        if self.negative:
            parts.append("-")
        if self.ns:
            parts.append(f"{self.ns}ns")
        if self.us:
            parts.append(f"{self.us}us")
        if self.ms:
            parts.append(f"{self.ms}ms")
        if self.s:
            parts.append(f"{self.s}s")
        if self.m:
            parts.append(f"{self.m}m")
        if self.h:
            parts.append(f"{self.h}h")
        if self.d:
            parts.append(f"{self.d}d")
        if self.w:
            parts.append(f"{self.w}w")
        if self.mo:
            parts.append(f"{self.mo}mo")
        if self.q:
            parts.append(f"{self.q}q")
        if self.y:
            parts.append(f"{self.y}y")

        result = "".join(parts)
        return result

    def is_zero(self) -> bool:
        return all((
            self.ns == 0,
            self.us == 0,
            self.ms == 0,
            self.s == 0,
            self.m == 0,
            self.h == 0,
            self.d == 0,
            self.w == 0,
            self.mo == 0,
            self.q == 0,
            self.y == 0,
        ))

    def _compare(self, other: "TimeDelta") -> int:
        if self.negative and not other.negative:
            return -1
        if not self.negative and other.negative:
            return 1
        if self.negative and other.negative:
            return -self._compare_positive(other)
        return self._compare_positive(other)

    def _compare_positive(self, other: "TimeDelta") -> int:
        for unit in ["y", "q", "mo", "w", "d", "h", "m", "s", "ms", "us", "ns"]:
            self_value = getattr(self, unit)
            other_value = getattr(other, unit)
            if self_value < other_value:
                return -1
            elif self_value > other_value:
                return 1
        return 0

    def __lt__(self, other: object) -> bool:
        if isinstance(other, str):
            other = parse_timedelta(other)
        if not isinstance(other, TimeDelta):
            return NotImplemented
        return self._compare(other) < 0

    def __gt__(self, other: object) -> bool:
        if isinstance(other, str):
            other = parse_timedelta(other)
        if not isinstance(other, TimeDelta):
            return NotImplemented
        return self._compare(other) > 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            other = parse_timedelta(other)
        if not isinstance(other, TimeDelta):
            return NotImplemented
        return self._compare(other) == 0

    def is_multiple_of(self, other: "TimeDelta") -> bool:
        if self.is_zero() != other.is_zero():
            return False
        elif self.is_zero() and other.is_zero():
            return True
        if self.negative != other.negative:
            return False
        for unit in ["y", "q", "mo", "w", "d", "h", "m", "s", "ms", "us", "ns"]:
            self_value = getattr(self, unit)
            other_value = getattr(other, unit)
            if other_value == 0:
                if self_value != 0:
                    return False
            elif self_value % other_value != 0:
                return False
        return True
        
def parse_timedelta_str(delta: str) -> TimeDelta:
    negative = False
    if delta.startswith("-"):
        negative = True
        delta = delta[1:]
    elif delta.startswith("+"):
        delta = delta[1:]

    pattern = re.compile(r"(?P<value>\d+)(?P<unit>ns|us|ms|s|m|h|d|w|mo|q|y)(?![a-z])")
    matches = pattern.findall(delta)

    td = TimeDelta(negative=negative)
    ns = 0
    us = 0
    ms = 0
    s = 0
    m = 0
    h = 0
    d = 0
    w = 0
    mo = 0
    q = 0
    y = 0
    for value, unit in matches:
        value = int(value)
        match unit:
            case "ns":
                ns += value
            case "us":
                us += value
            case "ms":
                ms += value
            case "s":
                s += value
            case "m":
                m += value
            case "h":
                h += value
            case "d":
                d += value
            case "w":
                w += value
            case "mo":
                mo += value
            case "q":
                q += value
            case "y":
                y += value

    if ns >= 1_000:
        us += ns // 1_000
        ns = ns % 1_000
    if us >= 1_000:
        ms += us // 1_000
        us = us % 1_000
    if ms >= 1_000:
        s += ms // 1_000
        ms = ms % 1_000
    if s >= 60:
        m += s // 60
        s = s % 60
    if m >= 60:
        h += m // 60
        m = m % 60
    if mo >= 3:
        q += mo // 3
        mo = mo % 3
    if q >= 4:
        y += q // 4
        q = q % 4
    # can't upcast hours to days, because days are a calander day which can be 23, 24 or 25 hours depending on daylight savings time
    # same for the rest of the units

    td = TimeDelta(ns=ns, us=us, ms=ms, s=s, m=m, h=h, d=d, w=w, mo=mo, q=q, y=y, negative=negative)

    return td

def parse_timedelta_timedelta(delta: timedelta) -> TimeDelta:
    total_seconds = delta.total_seconds()
    negative = total_seconds < 0
    total_seconds = abs(total_seconds)

    days = delta.days
    seconds = delta.seconds
    microseconds = delta.microseconds

    td = TimeDelta(
        d=days,
        s=seconds,
        ms=microseconds,
        negative=negative
    )

    return td

def parse_timedelta(delta: Optional[Union[str, timedelta, TimeDelta, "TimeDeltaOperator"]]) -> TimeDelta:
    if delta is None:
        return TimeDelta()
    elif isinstance(delta, str):
        if not _is_valid_polars_delta(delta):
            raise ValueError(f"Invalid time delta: {delta}")
        return parse_timedelta_str(delta)
    elif isinstance(delta, timedelta):
        return parse_timedelta_timedelta(delta)
    elif isinstance(delta, TimeDelta):
        return delta
    elif isinstance(delta, TimeDeltaOperator):
        return delta.td
    else:
        raise ValueError(f"Invalid time delta: {delta}")

def _is_valid_polars_delta(delta: str) -> bool:
    try:
        pl.DataFrame().select(pl.lit(0).cast(pl.Datetime).dt.offset_by(delta))
        return True
    except Exception:
        return False

# custom class to include month as a native time delta unit, since python's timedelta does not support months
class TimeDeltaOperator:
    def __init__(self, delta: Optional[Union[str, timedelta, TimeDelta, "TimeDeltaOperator"]] = None):
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