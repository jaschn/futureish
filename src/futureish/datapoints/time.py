from dataclasses import dataclass

from .base import DataPoint
from .._internal.constants import TIME

@dataclass(frozen=True)
class TimeDataPoint(DataPoint):
    """
    Data point that is based on the time column in the raw data. This data point is always known in the future.
    """
    def __init__(self):
        super().__init__(name=TIME, known_future=True)