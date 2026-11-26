#type: ignore
import calendar
from datetime import datetime

import numpy as np
import polars as pl

from futureish.datapoints.recorded import RecordedDataPoint
from futureish.utils.polars import add_contiguous_group, add_cos_sin_features


def test_add_contiguous_group_sorts_and_marks_gaps():
	frame = pl.DataFrame(
		{
			"time": pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 4), interval="1m", eager=True).gather([2, 0, 1, 4]),
			"value": [1, 2, 3, 4]
		}
	)
	result = add_contiguous_group(frame, [RecordedDataPoint(name="value", step_size="1m", known_future=True)])
	assert result["contiguous_group"].to_list() == [0, 0, 0, 1]


def test_add_cos_sin_features_matches_cyclic_formulas():
	times = [
		datetime(2024, 1, 1, 0, 0),
		datetime(2024, 1, 7, 0, 0),
		datetime(2024, 2, 29, 12, 0),
		datetime(2024, 7, 4, 6, 30),
		datetime(2024, 12, 1, 0, 0),
		datetime(2024, 12, 30, 0, 0),
		datetime(2024, 12, 31, 23, 59),
		datetime(2025, 1, 2, 12, 15),
	]
	frame = pl.DataFrame({"time": times})

	day_fraction = np.array([
		(time.hour + time.minute / 60 + time.second / 3600) / 24
		for time in times
	])
	hour = np.array([
		time.hour + time.minute / 60 + time.second / 3600
		for time in times
	])
	weekday = np.array([time.isoweekday() for time in times]) + day_fraction
	month = np.array([
		time.month + (time.day - 1) / calendar.monthrange(time.year, time.month)[1]
		for time in times
	]) + day_fraction
	ordinal = np.array([time.timetuple().tm_yday for time in times]) + day_fraction
	days_in_year = np.array([366 if calendar.isleap(time.year) else 365 for time in times])
	year = np.array([
		time.year + (day - 1) / year_length
		for time, day, year_length in zip(times, ordinal, days_in_year)
	])

	result = add_cos_sin_features(frame, year=True, year_minimum=year.min(), year_maximum=year.max())

	def expected(values, minimum, maximum):
		phase = (values - minimum) / (maximum - minimum) * 2 * np.pi
		return np.sin(phase), np.cos(phase)

	expected_features = {
		"hour": expected(hour, 0, 24),
		"weekday": expected(weekday, 1, 8),
		"month": expected(month, 1, 13),
		"ordinal_day": expected(ordinal, 1, 1 + days_in_year),
		"year": expected(year, year.min(), year.max()),
	}
	for name, (expected_sin, expected_cos) in expected_features.items():
		np.testing.assert_allclose(result[f"{name}_sin"].to_numpy(), expected_sin, rtol=1e-12, atol=1e-12)
		np.testing.assert_allclose(result[f"{name}_cos"].to_numpy(), expected_cos, rtol=1e-12, atol=1e-12)