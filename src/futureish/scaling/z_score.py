from typing import Any

import polars as pl

from ..utils.polars import AnyFrame, eager_row
from .data import DataScaler
from .._internal.constants import TIME_CL

class ZScoreScaler(DataScaler):
	def __init__(self, df: AnyFrame, raw_column_name: str, feature_column_name: str):
		super().__init__(df, raw_column_name, feature_column_name)
		self.mean, self.std = eager_row(df, self.raw_column.mean().alias("_mean"), self.raw_column.std().alias("_std"))
		if self.std == 0:
			self.std = 1 # just remove the scaling effect if the std is zero
		assert self.std != 0, f"Standard deviation of column '{self.raw_column_name}' is zero, cannot scale data."
	
	@property
	def _repr_fields(self) -> dict[str, Any]:
		new_fields: dict[str, Any] = {
			"mean": self.mean,
			"std": self.std,
		}

		return super()._repr_fields | new_fields

	def _scale(self, expr: pl.Expr, with_t0: bool = True) -> pl.Expr:
		return (expr - self.mean) / self.std

	def _inverse_scale(self, expr: pl.Expr) -> pl.Expr:
		anchor = pl.col("t0").max()
		no_anchor = anchor.is_null()
		return pl.concat([
			expr.filter(no_anchor | (TIME_CL <= anchor)),
			(expr.filter(~no_anchor & (TIME_CL > anchor)) * self.std + self.mean),
		])
