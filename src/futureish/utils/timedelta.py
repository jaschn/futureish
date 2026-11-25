from dataclasses import dataclass
from datetime import timedelta
from functools import total_ordering
from typing import Optional, Union

import polars as pl
import re

from ..utils.time_utils import parse_timedelta

UNIT_MAPPING = {
    "ns": {"bigger": ("us", 1_000), "smaller": None},
    "us": {"bigger": ("ms", 1_000), "smaller": ("ns", 1_000)},
    "ms": {"bigger": ("s", 1_000), "smaller": ("us", 1_000)},
    "s": {"bigger": ("m", 60), "smaller": ("ms", 1_000)},
    "m": {"bigger": ("h", 60), "smaller": ("s", 60)},
    "h": {"bigger": None, "smaller": ("m", 60)},
    "d": {"bigger": ("w", 7), "smaller": None}, # can be 23, 24 or 25 hours depending on daylight savings time
    "w": {"bigger": None, "smaller": ("d", 7)},
    "mo": {"bigger": ("q", 3), "smaller": None}, # can be 28, 29, 30 or 31 days depending on the month
    "q": {"bigger": ("y", 4), "smaller": (3, "mo")},
    "y": {"bigger": None, "smaller": (12, "mo")},
}

@total_ordering
@dataclass
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

    def _time_units(self):
        return [k for k in self.__dict__.keys() if k != "negative"]

    def __hash__(self):
        return hash(
            (self.ns, self.us, self.ms, self.s, self.m, self.h, self.d, self.w, self.mo, self.q, self.y, self.negative)
        )

    def __post_init__(self):
        self.normalize()

    def _borrow_from_higher_unit(self, unit: str):
        m = UNIT_MAPPING[unit]["bigger"]
        if m is None:
            raise ValueError(f"Cannot borrow from higher unit for {unit}")
        u, multiplier = m
        if getattr(self, u) <= 0:
            self._borrow_from_higher_unit(u) # go to a higher unit to borrow until the higher unit can't be converted to the lower unit anymore
        if getattr(self, u) > 0:
            setattr(self, u, getattr(self, u) - 1)
            setattr(self, unit, getattr(self, unit) + multiplier)
        else:
            raise ValueError(f"Cannot borrow from higher unit for {unit}")

    def _normalize_to_bigger_unit(self, unit: str):
        m = UNIT_MAPPING[unit]["bigger"]
        if m is None:
            return
        u, multiplier = m
        if getattr(self, unit) >= multiplier:
            setattr(self, u, getattr(self, u) + getattr(self, unit) // multiplier)
            setattr(self, unit, getattr(self, unit) % multiplier)

    def normalize(self):
        # Borrow from higher units if any unit is negative
        for unit in self._time_units():
            if getattr(self, unit) < 0:
                self._borrow_from_higher_unit(unit)

        for unit in self._time_units():
            self._normalize_to_bigger_unit(unit)

        if any(getattr(self, unit) < 0 for unit in self._time_units()):
            raise ValueError("TimeDelta units must be non-negative integers")

        if self.negative and self.is_zero():
            self.negative = False

    def is_ambiguous(self) -> bool:
        if self.h >= 23:
            if (
                self.d != 0 or 
                (self.h > 24 * 7 - 1 and self.w != 0) or
                (self.h > 24 * 28 - 1 and self.mo != 0) or
                (self.h > 24 * 3 * 30 - 1 and self.q != 0) or
                (self.h > 24 * 12 * 30 - 1 and self.y != 0)
            ):
                return True
        if self.w >= 4:
            if (
                self.mo != 0 or 
                (self.w > 7 * 30 - 1 and self.q != 0) or
                (self.w > 4 * 12 - 1 and self.y != 0)
            ):
                return True
        return False

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
        if len(result) == 0:
            return "0s"
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

    def __mul__(self, other: int) -> "TimeDelta":
        if not isinstance(other, int):
            return NotImplemented
        return TimeDelta(
            negative=self.negative,
            ns=self.ns * other,
            us=self.us * other,
            ms=self.ms * other,
            s=self.s * other,
            m=self.m * other,
            h=self.h * other,
            d=self.d * other,
            w=self.w * other,
            mo=self.mo * other,
            q=self.q * other,
            y=self.y * other,
        )

    def __add__(self, other: "TimeDelta") -> "TimeDelta":
        if not isinstance(other, TimeDelta):
            return NotImplemented
        if self.negative == other.negative:
            return TimeDelta(
                negative=self.negative,
                ns=self.ns + other.ns,
                us=self.us + other.us,
                ms=self.ms + other.ms,
                s=self.s + other.s,
                m=self.m + other.m,
                h=self.h + other.h,
                d=self.d + other.d,
                w=self.w + other.w,
                mo=self.mo + other.mo,
                q=self.q + other.q,
                y=self.y + other.y,
            )
        elif self.negative and not other.negative:
            if self.positive() > other:
                return TimeDelta(
                    negative=True,
                    ns=self.ns - other.ns,
                    us=self.us - other.us,
                    ms=self.ms - other.ms,
                    s=self.s - other.s,
                    m=self.m - other.m,
                    h=self.h - other.h,
                    d=self.d - other.d,
                    w=self.w - other.w,
                    mo=self.mo - other.mo,
                    q=self.q - other.q,
                    y=self.y - other.y,
                )
            else:
                return TimeDelta(
                    negative=False,
                    ns=other.ns - self.ns,
                    us=other.us - self.us,
                    ms=other.ms - self.ms,
                    s=other.s - self.s,
                    m=other.m - self.m,
                    h=other.h - self.h,
                    d=other.d - self.d,
                    w=other.w - self.w,
                    mo=other.mo - self.mo,
                    q=other.q - self.q,
                    y=other.y - self.y,
                )
        else:
            return NotImplemented

    def negated(self) -> "TimeDelta":
        return TimeDelta(
            negative=True,
            ns=self.ns,
            us=self.us,
            ms=self.ms,
            s=self.s,
            m=self.m,
            h=self.h,
            d=self.d,
            w=self.w,
            mo=self.mo,
            q=self.q,
            y=self.y,
        )

    def positive(self) -> "TimeDelta":
        return TimeDelta(
            negative=False,
            ns=self.ns,
            us=self.us,
            ms=self.ms,
            s=self.s,
            m=self.m,
            h=self.h,
            d=self.d,
            w=self.w,
            mo=self.mo,
            q=self.q,
            y=self.y,
        )

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
        
