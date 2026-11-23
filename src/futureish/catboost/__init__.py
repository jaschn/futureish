from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path
from .model import CatBoostTimeseriesModel, CatboostFitConfig

__path__ = extend_path(__path__, __name__)

try:
    __version__ = version("timeseries-catboost")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["CatBoostTimeseriesModel", "CatboostFitConfig"]