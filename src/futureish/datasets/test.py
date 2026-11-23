from .config import DataConfig
from .base import DatasetBase
from .predict import PredictDatasetBase
from .._internal.constants import CONTIGUOUS_GROUP, TRAJ_ID
from ..utils.polars import AnyFrame
from ..features.input_base import InputFeatureBase
from ..features.target import TargetFeature
from .._internal.constants import TIME_CL


import polars as pl


from datetime import datetime
from typing import Optional, Sequence


class TestDataset(PredictDatasetBase):
	__test__ = False # avoids pytest from trying to run this class as a test case
	# expects_future_data_known: bool = False # TODO: check this
	pass