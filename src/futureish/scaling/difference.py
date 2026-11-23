import polars as pl

from ..utils.polars import AnyFrame
from .data import DataScaler
from .._internal.constants import TIME_CL


class DiffScaler(DataScaler):
	def __init__(self, df: AnyFrame, raw_column_name: str, feature_column_name: str):
		super().__init__(df, raw_column_name, feature_column_name)

	def _scale(self, expr: pl.Expr, with_t0: bool = True) -> pl.Expr:
		if with_t0:
			# no_anchor: a group (e.g. one trajectory of a batched recursive
			# predict) can have no "t0" at all - already past its own last
			# step while other groups in the same call keep going. Without
			# this, both filters below come up empty for that group (every
			# comparison against a null anchor is null) and pl.concat()
			# raises a shape mismatch instead of leaving the group untouched.
			anchor = pl.col("t0").max()
			no_anchor = anchor.is_null()
			return pl.concat([
					expr.filter(no_anchor | (TIME_CL <= anchor)),
					expr.diff().filter(~no_anchor & (TIME_CL > anchor)),
				]
			)
		else:
			return expr.diff()

	def _inverse_scale(self, expr: pl.Expr) -> pl.Expr:
		# See _scale()'s no_anchor comment - same reasoning here.
		anchor = pl.col("t0").max()
		no_anchor = anchor.is_null()
		return pl.concat([
			expr.filter(no_anchor | (TIME_CL < anchor)),
			expr.filter(~no_anchor & (TIME_CL >= anchor)).cum_sum(),
		])
