from dataclasses import dataclass, field
from typing import Any
import polars as pl

from .._internal.repr import ReprMixin

@dataclass(frozen=True)
class DataPoint(ReprMixin):
	"""
	Base class for a data point in the raw data. Defines the name and if the data point is known in the future.
	"""
	name: str
	known_future: bool = field(default=False, kw_only=True)

	@property
	def column(self) -> pl.Expr:
		return pl.col(self.name)
	
	@property
	def _repr_fields(self) -> dict[str, Any]:
		return {
			"name": self.name,
			"known_future": self.known_future,
		}