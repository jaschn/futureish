from importlib.metadata import PackageNotFoundError, version

try:
	__version__ = version("timeseries-gpr")
except PackageNotFoundError:
	__version__ = "0.0.0+unknown"

from .gpytorch import (
	GPyTorchFitConfig,
	GPyTorchTrainer,
	GPyTorchOnlinePEP,
	GPyTorchSGP,
	GPyTorchVGP,
)

__all__ = [
	"GPyTorchFitConfig",
	"GPyTorchTrainer",
	"GPyTorchOnlinePEP",
	"GPyTorchSGP",
	"GPyTorchVGP",
]