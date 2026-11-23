from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class FitConfig:
	"""Base class for a model backend's fit()/fit_step() configuration.

	ABCModel.fit()/fit_step() take a single `config` parameter typed as
	FitConfig so every backend can be driven through the same generic
	MultiOutputAssemblyModel/Forecaster call, regardless of what it needs to
	configure. Each backend that has fit() parameters defines its own frozen
	dataclass subclassing this one (e.g. GPyTorchFitConfig) with exactly those
	parameters as typed, defaulted fields, and declares that subclass as the
	`config` type on its own fit()/fit_step() override - so constructing a
	config gets full autocomplete/type-checking, even though the abstract
	signature here only promises a plain FitConfig. Backends with nothing to
	configure (e.g. DummyModel) can ignore `config` entirely.
	"""

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)