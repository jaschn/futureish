#type: ignore
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
		datetime(2024, 7, 4, 6, 30),
		datetime(2025, 1, 2, 12, 15),
	]
	frame = pl.DataFrame({"time": times})

	hour = np.array([0.0, 6.5, 12.25])
	weekday = np.array([1.0, 4.0, 4.0]) + np.array([0.0, 6.5 / 24, 12.25 / 24])
	ordinal = np.array([1.0, 186.0, 2.0]) + np.array([0.0, 6.5 / 24, 12.25 / 24])
	year = np.array([2024.0, 2024.0 + 185 / 366 + 6.5 / (24 * 366), 2025.0 + 1 / 365 + 12.25 / (24 * 365)])

	result = add_cos_sin_features(frame, year=True, year_minimum=year.min(), year_maximum=year.max())

	def expected(values, minimum, maximum):
		phase = (values - minimum) / (maximum - minimum) * 2 * np.pi
		return np.sin(phase), np.cos(phase)

	expected_features = {
		"hour": expected(hour, 0, 24),
		"weekday": expected(weekday, 1, 7),
		"ordinal_day": expected(ordinal, 1, 365),
		"year": expected(year, year.min(), year.max()),
	}
	for name, (expected_sin, expected_cos) in expected_features.items():
		np.testing.assert_allclose(result[f"{name}_sin"].to_numpy(), expected_sin, rtol=1e-12, atol=1e-12)
		np.testing.assert_allclose(result[f"{name}_cos"].to_numpy(), expected_cos, rtol=1e-12, atol=1e-12)