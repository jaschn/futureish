import abc
from typing import Callable, TypeAlias, Union, Optional

import polars as pl

from ..utils.polars import AnyFrame, FrameT, frame_columns
from .base import ScalerBase
from .._internal.constants import TIME_CL

class DataScaler(ScalerBase, abc.ABC):
	def scale(self, df: FrameT, over: Optional[Union[str, list[str]]] = None, with_t0: Optional[bool] = None) -> FrameT:
		if with_t0 is None:
			with_t0 = "t0" in frame_columns(df)
		scale_expr = self._scale(self.raw_column, with_t0=with_t0)
		if over is not None and len(over) > 0:
			scale_expr = scale_expr.over(over)

		return (
			df
			.drop_nulls(["time", self.raw_column_name])
			.sort(TIME_CL)
			.with_columns(
					scale_expr.alias(self.feature_column_name)
				)
			.select(
					"time",
					pl.exclude(self.raw_column_name, "time") # make sure time is always there, even if the raw column is the time columns
				)
			)

	def inverse_scale(self, df: FrameT, over: Optional[Union[str, list[str]]] = None) -> FrameT:
		expr = self._inverse_scale(self.column)
		if over is not None and len(over) > 0:
			expr = expr.over(over)

		return (
				df
				.sort(TIME_CL)
				.with_columns(
						expr.alias(self.raw_column_name)
				)
				.select(pl.exclude(self.feature_column_name))
			)

	@abc.abstractmethod
	def _scale(self, expr: pl.Expr, with_t0: bool = True) -> pl.Expr:
		raise NotImplementedError("DataScaler._scale() must be implemented in subclasses")

	@abc.abstractmethod
	def _inverse_scale(self, expr: pl.Expr) -> pl.Expr:
		raise NotImplementedError("DataScaler._inverse_scale() must be implemented in subclasses")


ScalerFactory: TypeAlias = Callable[[AnyFrame, str, str], ScalerBase]
