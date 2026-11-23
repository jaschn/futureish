from typing import Optional, Union

import polars as pl

from ..utils.polars import AnyFrame, FrameT
from .._internal.repr import ReprMixin


class ScalerBase(ReprMixin):
	"""Column-name bookkeeping shared by every scaler, leaf or composite.

	DataScaler (simple, elementwise scaling) and ChainedScaler (composing
	several scaling stage-by-stage) both build on this for the raw/feature
	column names and the pl.Expr accessors built from them.
	"""

	def __init__(self, df: AnyFrame, raw_column_name: str, feature_column_name: str):
		self.raw_column_name = raw_column_name
		self.feature_column_name = feature_column_name

	@property
	def _repr_fields(self) -> dict[str, str]:
		return {
			"raw_column_name": self.raw_column_name,
			"feature_column_name": self.feature_column_name,
		}

	@property
	def column(self) -> pl.Expr:
		return pl.col(self.feature_column_name)

	@property
	def raw_column(self) -> pl.Expr:
		return pl.col(self.raw_column_name)

	def scale(self, df: FrameT, over: Optional[Union[str, list[str]]] = None, with_t0: Optional[bool] = None) -> FrameT: ...

	def inverse_scale(self, df: FrameT, over: Optional[Union[str, list[str]]] = None) -> FrameT: ...
