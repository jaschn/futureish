from futureish.utils.timedeltaoperator import TimeDeltaOperator
from futureish.utils.timedelta import TimeDelta
import pytest

base_time_units = ["ns", "us", "ms", "s", "m", "h", "d", "w", "mo", "q", "y"]
possible_base_time_units_combination = [
    str(i) + unit for unit in base_time_units
    for i in range(1, 3)
] + [
    str(ii) + base_time_units[i-1] + str(ii) + base_time_units[i]
    for i in range(1, len(base_time_units))
    for ii in range(1, 3)
]

@pytest.mark.parametrize(
    "delta",
    possible_base_time_units_combination
)
def test_parse_timedelta(delta: str):
    operator = TimeDeltaOperator(delta)
    assert operator.td.to_string() == delta, f"Expected {delta}, but got {operator.td.to_string()} with parsed values: {operator.td.__dict__}"
    operator2 = TimeDeltaOperator(operator.td.to_string())
    assert operator == operator2, f"Expected {operator}, but got {operator2} with parsed values: {operator2.td.__dict__}"

def test_parse_timedelta_invalid():
    with pytest.raises(ValueError):
        TimeDeltaOperator("invalid")

def test_less_than_or_equal():
    assert TimeDelta(1) <= TimeDelta(1)
    assert TimeDelta(1) <= TimeDelta(2)
    assert TimeDeltaOperator("1s") <= TimeDeltaOperator("1s")
    assert TimeDeltaOperator("1s") <= TimeDeltaOperator("2s")

@pytest.mark.parametrize(
    "value, unit, next_unit",
    [
        [1000, "ns", "us"],
        [1000, "us", "ms"],
        [1000, "ms", "s"],
        [60, "s", "m"],
        [60, "m", "h"],
    ]
)
def test_parse_timedelta_upcast(value: int, unit: str, next_unit: str):
    delta = f"{value}{unit}"
    operator = TimeDeltaOperator(delta)
    assert getattr(operator.td, unit) == 0, f"Expected {unit} to be 0 after upcasting, but got {getattr(operator.td, unit)}"
    assert getattr(operator.td, next_unit) == 1, f"Expected {next_unit} to be 1 after upcasting, but got {getattr(operator.td, next_unit)}"