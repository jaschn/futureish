from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from futureish.utils.timedelta import TimeDelta
    from futureish.utils.timedeltaoperator import TimeDeltaOperator
import polars as pl
import re

from datetime import timedelta
from typing import Optional, Union

def parse_timedelta_str(delta: str) -> "TimeDelta":
    from futureish.utils.timedelta import TimeDelta
    negative = False
    if delta.startswith("-"):
        negative = True
        delta = delta[1:]
    elif delta.startswith("+"):
        delta = delta[1:]

    pattern = re.compile(r"(?P<value>\d+)(?P<unit>ns|us|ms|s|m|h|d|w|mo|q|y)(?![a-z])")
    matches = pattern.findall(delta)

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

    td = TimeDelta(ns=ns, us=us, ms=ms, s=s, m=m, h=h, d=d, w=w, mo=mo, q=q, y=y, negative=negative)

    return td

def parse_timedelta_timedelta(delta: timedelta) -> "TimeDelta":
    from futureish.utils.timedelta import TimeDelta
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

def _is_valid_polars_delta(delta: str) -> bool:
    try:
        pl.DataFrame().select(pl.lit(0).cast(pl.Datetime).dt.offset_by(delta))
        return True
    except Exception:
        return False

def parse_timedelta(delta: Optional[Union[str, timedelta, "TimeDelta", "TimeDeltaOperator"]]) -> "TimeDelta":
    from futureish.utils.timedelta import TimeDelta
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
    else:
        raise ValueError(f"Invalid time delta: {delta}")