from dataclasses import dataclass, field
from typing import Any

from ..utils.timedeltaoperator import TimeDeltaOperator

from .base import DataPoint

@dataclass(frozen=True)
class RecordedDataPoint(DataPoint):
	"""
	Data point for a recorded value in the raw data.
	"""
	step_size: TimeDeltaOperator
	def __init__(self, name: str, step_size: str, known_future: bool = False):
		super().__init__(name=name, known_future=known_future)
		object.__setattr__(self, "step_size", TimeDeltaOperator(step_size))

	@property
	def _repr_fields(self) -> dict[str, Any]:
		new_fields = {
			"step_size": self.step_size,
		}
		return super()._repr_fields | new_fields