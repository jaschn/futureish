from typing import Optional, Union

import polars as pl

from .._internal.constants import TIME_CL
from .base import ScalerBase
from ..utils.polars import FrameT

from .data import DataScaler


class NoopScaler(ScalerBase):
	def scale(self, df: FrameT, over: Optional[Union[str, list[str]]] = None, with_t0: Optional[bool] = None) -> FrameT:
		return (
			df
			.drop_nulls(["time", self.raw_column_name])
			.sort(TIME_CL)
			.with_columns(
					self.raw_column.alias(self.feature_column_name)
				)
			.select(
					"time",
					pl.exclude(self.raw_column_name, "time") # make sure time is always there, even if the raw column is the time columns
				)
			)
	
	def inverse_scale(self, df: FrameT, over: Optional[Union[str, list[str]]] = None, with_t0: Optional[bool] = None) -> FrameT:
		return (
				df
				.sort(TIME_CL)
				.with_columns(
						self.column.alias(self.raw_column_name)
				)
				.select(pl.exclude(self.feature_column_name))
			)