from datetime import datetime
import functools
import warnings
from typing import Any, Literal, Optional, Sequence, TypeAlias, TypeVar, Union, cast
import numpy as np
import polars as pl

from ..utils.timedelta import TimeDeltaOperator

from ..datapoints.base import DataPoint
from ..datapoints.time import TimeDataPoint

from .._internal.constants import CONTIGUOUS_GROUP, TIME_CL

# Constrained (not bound) so a function like `def f(df: FrameT) -> FrameT`
# ties its return type to whichever concrete type the caller passed in -
# passing a LazyFrame through the data-processing pipeline keeps it typed as
# a LazyFrame all the way, passing a DataFrame keeps it a DataFrame.
FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)

# Plain union for signatures that accept either frame type but don't need to
# tie an output type to it (constructors, container element types, anything
# that doesn't return a frame of the same kind it was handed).
AnyFrame: TypeAlias = Union[pl.DataFrame, pl.LazyFrame]


def frame_columns(frame: AnyFrame) -> list[str]:
	"""Column names for `frame`, whether it's eager or lazy.

	LazyFrame.columns resolves (and warns about resolving) the schema on
	every access; collect_schema().names() does the same resolution without
	the warning and works identically on an eager DataFrame, so callers that
	may see either frame type during a lazy prediction pipeline can use this
	instead of branching on the frame type.
	"""
	return frame.collect_schema().names()


def eager_row(frame: AnyFrame, *exprs: pl.Expr) -> tuple[Any, ...]:
	"""Evaluate scalar expressions against `frame` and return them as a row,
	collecting only if `frame` is still lazy.

	Lets code that needs a concrete value for control flow (a boundary
	timestamp, a validation check) pull it out of a lazy pipeline without
	forcing every caller to know whether the frame it was handed is lazy or
	already eager.
	"""
	selected = frame.select(*exprs)
	if isinstance(selected, pl.LazyFrame):
		selected = selected.collect()
	return selected.row(0)

def to_eager(frame: AnyFrame) -> pl.DataFrame:
	"""Collect `frame` if it's still lazy, otherwise return it unchanged.

	The natural place for a lazy pipeline to end: wherever a caller needs a
	real DataFrame (e.g. to convert to numpy), regardless of whether it was
	handed a DataFrame or a LazyFrame.
	"""
	return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def eager_item(frame: AnyFrame, expr: pl.Expr) -> Any:
	"""Evaluate a scalar expression against `frame` and return it as a single
	value, collecting only if `frame` is still lazy.

	Lets code that needs a concrete value for control flow (a boundary
	timestamp, a validation check) pull it out of a lazy pipeline without
	forcing every caller to know whether the frame it was handed is lazy or
	already eager.
	"""
	return eager_row(frame, expr)[0]

def _cos_transform(col: Union[pl.Expr, str], minimum: Optional[Union[float, pl.Expr]] = None, maximum: Optional[Union[float, pl.Expr]] = None) -> pl.Expr:
	if isinstance(col, str):
		col = pl.col(col)
	if minimum is None:
		minimum = col.min()
	if maximum is None:
		maximum = col.max()
	return (((col - minimum) / (maximum - minimum)) * np.pi * 2).cos()

def _sin_transform(col: Union[pl.Expr, str], minimum: Optional[Union[float, pl.Expr]] = None, maximum: Optional[Union[float, pl.Expr]] = None) -> pl.Expr:
	if isinstance(col, str):
		col = pl.col(col)
	if minimum is None:
		minimum = col.min()
	if maximum is None:
		maximum = col.max()
	return (((col - minimum) / (maximum - minimum)) * np.pi * 2).sin()

def add_cos_sin_features(df: FrameT, time_column: Optional[Union[pl.Expr, str]] = None, *, second: bool = True, minute: bool = True, hour: bool = True, weekday: bool = True, month: bool = True, ordinal_day: bool = True, year: bool = False, year_minimum: Optional[float] = None, year_maximum: Optional[float] = None) -> FrameT:

	if time_column is None:
		time_column = "time"
	if isinstance(time_column, str):
		time_column = pl.col(time_column)

	second_float = time_column.dt.second(fractional=True)
	minute_float = time_column.dt.minute() + second_float / 60.0
	hour_float = (time_column.dt.hour() + minute_float / 60.0)
	weekday_float = (time_column.dt.weekday() + hour_float / 24.0)
	month_float = (time_column.dt.month() + (time_column.dt.day() - 1) / time_column.dt.days_in_month() + hour_float / 24.0)
	ordinal_day_float = (time_column.dt.ordinal_day() + hour_float / 24.0)
	year_float = time_column.dt.year() + (ordinal_day_float - 1) / (365.0 + time_column.dt.is_leap_year().cast(pl.Float64)) 
	df_added = df

	if second:
		df_added = df_added.with_columns(
				_sin_transform(second_float, minimum=0, maximum=60).alias("second_sin"),
				_cos_transform(second_float, minimum=0, maximum=60).alias("second_cos"),
		)
	if minute:
		df_added = df_added.with_columns(
				_sin_transform(minute_float, minimum=0, maximum=60).alias("minute_sin"),
				_cos_transform(minute_float, minimum=0, maximum=60).alias("minute_cos"),
		)
	if hour:
		df_added = df_added.with_columns(
				_sin_transform(hour_float, minimum=0, maximum=24).alias("hour_sin"),
				_cos_transform(hour_float, minimum=0, maximum=24).alias("hour_cos"),
		)
	if weekday:
		df_added = df_added.with_columns(
				_sin_transform(weekday_float, minimum=1, maximum=7).alias("weekday_sin"),
				_cos_transform(weekday_float, minimum=1, maximum=7).alias("weekday_cos"),
		)
	if month:
		df_added = df_added.with_columns(
				_sin_transform(month_float, minimum=1, maximum=12).alias("month_sin"),
				_cos_transform(month_float, minimum=1, maximum=12).alias("month_cos"),
		)
	if ordinal_day:
		df_added = df_added.with_columns(
				_sin_transform(ordinal_day_float, minimum=1, maximum=365).alias("ordinal_day_sin"),
				_cos_transform(ordinal_day_float, minimum=1, maximum=365).alias("ordinal_day_cos"),
		)
	if year:
		if year_minimum is None:
			raise ValueError("year_minimum must be provided when year=True")
		if year_maximum is None:
			raise ValueError("year_maximum must be provided when year=True")
		df_added = df_added.with_columns(
				_cos_transform(year_float, minimum=year_minimum, maximum=year_maximum).alias("year_cos"),
				_sin_transform(year_float, minimum=year_minimum, maximum=year_maximum).alias("year_sin"),
		)
	return df_added

def _create_contiguous_group_feature(df: FrameT, base_datapoint: DataPoint) -> FrameT:
	from ..datapoints import RecordedDataPoint
	if isinstance(base_datapoint, TimeDataPoint):
		return (
			df
			.select("time")
			.with_columns(
				pl.lit(0).alias(CONTIGUOUS_GROUP + "_" + base_datapoint.name)
			)
		)

	assert isinstance(base_datapoint, RecordedDataPoint), "base_datapoint must be an instance of RecordedDataPoint"

	time_grp = (
		df
		.select(
			"time",
			base_datapoint.name
		)
		.drop_nulls()
		.drop(base_datapoint.name)
	)

	start: Optional[datetime] = eager_item(time_grp, base_datapoint.step_size.truncate_column(pl.col("time").min()))
	end: Optional[datetime] = eager_item(time_grp, pl.col("time").max())
	dtype = time_grp.collect_schema()["time"]
	assert isinstance(dtype, pl.Datetime), "time column must be of datetime type"

	if start is None or end is None:
		return df.select("time").with_columns(pl.int_range(pl.len()).alias(CONTIGUOUS_GROUP + "_" + base_datapoint.name))

	possible_times = (
		pl.LazyFrame()
		.select(
			pl.datetime_range(
						start=start,
						end=end,
						interval=base_datapoint.step_size.to_string(),
						time_unit=dtype.time_unit,
						time_zone=dtype.time_zone
					).alias("time")
			)
	)

	if isinstance(time_grp, pl.DataFrame):
		times_joined = (
			time_grp
			.join(
				possible_times.collect(),
				on="time",
				how="left",
				coalesce=False
			)
		)
	else:
		times_joined = (
			time_grp
			.join(
				possible_times,
				on="time",
				how="left",
				coalesce=False
			)
		)

	time_grp = (	
		times_joined
		.drop("time")
		.rename({"time_right": "time"})
		.sort(TIME_CL)
		.with_columns(
			(TIME_CL.shift(1) != base_datapoint.step_size.offset_column_backward(TIME_CL))
			.backward_fill()
			.cum_sum()
			.alias(CONTIGUOUS_GROUP + "_" + base_datapoint.name)
		)
	)

	return time_grp

def add_contiguous_group(df: FrameT, recorded_datapoints: Sequence[DataPoint]) -> FrameT:
	dfs: list[FrameT] = [df.select("time")]
	recorded_datapoints_set = set(recorded_datapoints)  # Remove duplicates
	for feature in recorded_datapoints_set:
		df_grp = _create_contiguous_group_feature(df, feature)
		dfs.append(df_grp)

	df_reduced = functools.reduce(lambda left, right: left.join(right, on="time", how="left"), dfs)
	df_reduced = df_reduced.sort("time").with_columns(
		pl.col("^" + CONTIGUOUS_GROUP + "_.*$").forward_fill()
	)

	df_reduced = df_reduced.sort("time").select(
		"time",
		pl.struct(pl.col("^" + CONTIGUOUS_GROUP + "_.*$")).rle_id().alias(CONTIGUOUS_GROUP)
	)

	return df.join(cast(FrameT, df_reduced), on="time", how="left")