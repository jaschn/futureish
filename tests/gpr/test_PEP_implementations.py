from datetime import datetime
from pathlib import Path
import pickle
from typing import cast
from futureish.forecasting.forecaster import Forecaster
import pytest
import polars as pl

from futureish.datasets.online import OnlineDataset
from futureish.datasets.config import DataConfig
from futureish.datasets.predict import PredictDatasetBase
from futureish.datasets.train import TrainDataset
from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchFitConfig

try:
    with open(Path(__file__).parent.parent / "test_data" / Path("test_cases.pkl"), "rb") as f:
        test_cases = pickle.load(f)
except FileNotFoundError:
    import os
    raise FileNotFoundError("test_cases.pkl not found. Current directory: " + str(Path.cwd()))

@pytest.fixture
def taxi_data() -> pl.DataFrame:
    base = Path(__file__).parent.parent / "test_data"
    taxi_data_file = base / "taxi_data.parquet"
    if not taxi_data_file.exists():
        import urllib.request
        base.mkdir(parents=True, exist_ok=True)
        pl.read_csv("https://raw.githubusercontent.com/unit8co/darts/refs/heads/master/datasets/taxi_new_york_passengers.csv", try_parse_dates=True).rename({"#Passengers": "value"}).write_parquet(taxi_data_file)
    return pl.read_parquet(Path(__file__).parent.parent / "test_data" / "taxi_data.parquet")

@pytest.mark.parametrize("case", [pytest.param(case, id=case['name']) for case in test_cases])
def test_maximum_likelihood(case):
    from futureish.gpr.gpytorch.backends.online_pep.sgpr_pep import SGPR_PEP
    import torch
    import gpytorch
    import numpy as np
    import copy

    X = torch.tensor(case['X'], dtype=torch.float64)
    Y = torch.tensor(case['Y'], dtype=torch.float64)
    Z = torch.tensor(case['Z'], dtype=torch.float64)

    kernel_lengthscale = case['lengthscales']
    kernel_scale = case['kernel_scale']
    variance = case['variance']

    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1])
    kernel.lengthscale = kernel_lengthscale
    covar_module = gpytorch.kernels.ScaleKernel(kernel)
    covar_module.outputscale = kernel_scale

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(variance, dtype=torch.float64)

    model = SGPR_PEP(X, Y, kernel=covar_module, likelihood=likelihood, z=Z, verify=True)
    l = model._maximum_log_likelihood_objective()

    assert pytest.approx(case['maximum_log_likelihood'], rel=1e-5) == l.item(), f"Expected {case['maximum_log_likelihood']}, but got {l.item()}"

    X_pred = torch.tensor(case['X_pred'], dtype=torch.float64)
    pred = model(X_pred)

    assert pytest.approx(pred.mean.detach().numpy(), rel=1e-5) == np.array(case['pred_mean']), f"Expected {case['pred_mean']}, but got {pred.mean.detach().numpy()}"
    assert pytest.approx(pred.covariance_matrix.detach().numpy(), rel=1e-5) == np.array(case['pred_cov']), f"Expected {case['pred_cov']}, but got {pred.covariance_matrix.detach().numpy()}"

    from futureish.gpr.gpytorch.backends.online_pep.osgpr_pep import OSGPR_PEP
    Zopt = model.inducing_points.detach().clone()
    pseudo = model(Zopt)
    mu = pseudo.mean.unsqueeze(-1).detach().clone()
    Su = pseudo.covariance_matrix.detach().clone()
    if Su.ndim == 3:
        Su = Su[0]
    assert Su.ndim == 2, "Su should be a 2D array"
    Kaa = model.covar_module(Zopt).detach().clone()
    Zinit = Zopt.detach().clone()
    kern = copy.deepcopy(model.covar_module)
    lk = copy.deepcopy(model.likelihood)
    alpha = model.alpha
    m_online = OSGPR_PEP(X, Y, kern, mu, Su, Kaa, Zopt, Zinit, alpha, lk, verify=True) # type: ignore
    online_l = m_online._maximum_log_likelihood_objective()

    assert pytest.approx(online_l.item(), rel=1e-2) == case["online_likelihood"], f"Expected {case['online_likelihood']}, but got {online_l.item()}"

    pred_online = m_online(X_pred)

    assert pytest.approx(pred_online.mean.detach().numpy(), rel=1e-3, abs=1e-3) == np.array(case['online_predictions_mean']), f"Expected {case['online_predictions_mean']}, but got {pred_online.mean.detach().numpy()}"
    assert pytest.approx(pred_online.covariance_matrix.detach().numpy(), rel=1e-2) == np.array(case['online_predictions_cov']), f"Expected {case['online_predictions_cov']}, but got {pred_online.covariance_matrix.detach().numpy()}"


@pytest.mark.parametrize("case", [pytest.param(case, id=case['name']) for case in test_cases])
def test_backward_pass(case):
    from futureish.gpr.gpytorch.backends.online_pep.sgpr_pep import SGPR_PEP
    import torch
    import gpytorch
    import copy

    X = torch.tensor(case['X'], dtype=torch.float64)
    Y = torch.tensor(case['Y'], dtype=torch.float64)
    Z = torch.tensor(case['Z'], dtype=torch.float64)

    kernel_lengthscale = case['lengthscales']
    kernel_scale = case['kernel_scale']
    variance = case['variance']

    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1])
    kernel.lengthscale = kernel_lengthscale
    covar_module = gpytorch.kernels.ScaleKernel(kernel)
    covar_module.outputscale = kernel_scale

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(variance, dtype=torch.float64)

    model = SGPR_PEP(X, Y, kernel=covar_module, likelihood=likelihood, z=Z, verify=True)

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(5):
        opt.zero_grad()
        loss = model.objective()
        loss.backward(retain_graph=True)
        opt.step()

    from futureish.gpr.gpytorch.backends.online_pep.osgpr_pep import OSGPR_PEP
    Zopt = model.inducing_points.detach().clone()
    pseudo = model(Zopt)
    mu = pseudo.mean.unsqueeze(-1).detach().clone()
    Su = pseudo.covariance_matrix.detach().clone()
    if Su.ndim == 3:
        Su = Su[0]
    assert Su.ndim == 2, "Su should be a 2D array"
    Kaa = model.covar_module(Zopt).detach().clone()
    Zinit = Zopt.detach().clone()
    kern = copy.deepcopy(model.covar_module)
    lk = copy.deepcopy(model.likelihood)
    alpha = model.alpha
    m_online = OSGPR_PEP(X, Y, kern, mu, Su, Kaa, Zopt, Zinit, alpha, lk, verify=True) # type: ignore

    opt = torch.optim.Adam(m_online.parameters(), lr=0.01)
    for i in range(5):
        opt.zero_grad()
        loss = m_online.objective()
        loss.backward(retain_graph=True)
        opt.step()

@pytest.fixture
def data_prep(taxi_data: pl.DataFrame) -> pl.DataFrame:
    import polars as pl
    from futureish.utils.polars import add_cos_sin_features

    data_prep = taxi_data.pipe(add_cos_sin_features)
    return data_prep

@pytest.fixture
def dataset(data_prep: pl.DataFrame) -> tuple[TrainDataset, PredictDatasetBase]:
    import polars as pl

    from futureish.scaling import DiffScaler, ChainedScaler, ZScoreScaler
    from futureish.datasets.online import OnlineDataset
    from futureish.features.recorded import RecordedInputFeature
    from futureish.datapoints.recorded import RecordedDataPoint
    from futureish.features.target import TargetFeature
    from futureish.utils.polars import add_cos_sin_features
    from datetime import timedelta

    value = RecordedDataPoint(name="value", known_future=False, step_size="30m")

    hour_cos = RecordedDataPoint(name="hour_cos", known_future=True, step_size="30m")
    hour_sin = RecordedDataPoint(name="hour_sin", known_future=True, step_size="30m")

    weekday_cos = RecordedDataPoint(name="weekday_cos", known_future=True, step_size="30m")
    weekday_sin = RecordedDataPoint(name="weekday_sin", known_future=True, step_size="30m")

    ordinal_day_cos = RecordedDataPoint(name="ordinal_day_cos", known_future=True, step_size="30m")
    ordinal_day_sin = RecordedDataPoint(name="ordinal_day_sin", known_future=True, step_size="30m")

    input_features = [
        RecordedInputFeature(base_datapoint=hour_cos, feature_name="hour_cos_feature", input_length="0m", uses_future_data=False),
        RecordedInputFeature(base_datapoint=hour_sin, feature_name="hour_sin_feature", input_length="0m", uses_future_data=False),
        RecordedInputFeature(base_datapoint=weekday_cos, feature_name="weekday_cos_feature", input_length="0m", uses_future_data=False),
        RecordedInputFeature(base_datapoint=weekday_sin, feature_name="weekday_sin_feature", input_length="0m", uses_future_data=False),
        RecordedInputFeature(base_datapoint=ordinal_day_cos, feature_name="ordinal_day_cos_feature", input_length="0m", uses_future_data=False),
        RecordedInputFeature(base_datapoint=ordinal_day_sin, feature_name="ordinal_day_sin_feature", input_length="0m", uses_future_data=False),
    ]

    target_features = [
        TargetFeature(base_datapoint=value, feature_name="target_value", scaler_factory=ChainedScaler.from_scalers(DiffScaler, ZScoreScaler))
    ]

    # The online window and split are both expressed as a fraction of the
    # data's actual time span (rather than a row-count fraction) since this
    # dataset's rows aren't perfectly evenly spaced - window < split so the
    # scaler is fit only on data available before the training/predict split.
    time_min = cast(datetime, data_prep["time"].min())
    time_max = cast(datetime, data_prep["time"].max())
    total_seconds = (time_max - time_min).total_seconds()
    window = f"{int(total_seconds * 0.55)}s"
    split = f"{int(total_seconds * 0.6)}s"

    online_dataset = OnlineDataset(raw_data=data_prep, input_features=input_features, target_features=target_features, size_of_first_online_window=window)
    train_dataset, predict_dataset = online_dataset.split_training_and_test_data(split)

    return train_dataset, predict_dataset

def test_full_model(dataset: tuple[TrainDataset, PredictDatasetBase], data_prep: pl.DataFrame, ):
    from futureish.scaling import DiffScaler, ChainedScaler, ZScoreScaler
    from futureish.features.recorded import RecordedInputFeature
    from futureish.datapoints.recorded import RecordedDataPoint
    from futureish.features.target import TargetFeature
    from futureish.utils.polars import add_cos_sin_features
    from datetime import timedelta
    from futureish.forecasting.forecaster import Forecaster

    train_dataset, predict_dataset = dataset

    training_data = train_dataset.create_training_data()
    training_splits: list[TrainDataset] = []
    l = len(training_data)
    remaining = training_data

    s = (l // 10)
    for i in range(0, 10):
        split, remaining = train_dataset.split(s)
        training_splits.append(split)

    from futureish.gpr.gpytorch.backends.online_pep.gpytorch_online_pep import GPyTorchOnlinePEP
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer
    import gpytorch
    import torch
    torch.set_default_dtype(torch.float64)

    def kernel_factory(batch_shape, x_dims, X, Y):
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=x_dims, batch_shape=batch_shape)
        kernel.lengthscale = torch.tensor(1.0)
        covar_module = gpytorch.kernels.ScaleKernel(kernel, batch_shape=batch_shape)
        covar_module.outputscale = torch.tensor(1.0)
        return covar_module

    model = Forecaster(GPyTorchTrainer(induce_size=20, kernel_factory=kernel_factory, basemodel_class=GPyTorchOnlinePEP))

    for split in training_splits:
        model.fit_step(split, config=GPyTorchFitConfig(iters=10))

    forecaster = Forecaster.from_fitted(model.model, train_dataset.config)

    start = predict_dataset.valid_t0.select(pl.col("t0").min()).item()

    pred_data = (
        data_prep
        .filter(pl.col("time") < start + timedelta(days=2), pl.col("time") >= start - timedelta(hours=1))
        .with_columns(
            pl.when(pl.col("time") < start).then(pl.col("value")).otherwise(None),
            pl.when(pl.col("time").dt.offset_by("30m") == start).then(pl.col("time")).otherwise(None).alias("t0")
        )
    )

    pred = forecaster.predict_autoregressive(pred_data, samples=100)

def test_online_model(dataset: tuple[TrainDataset, PredictDatasetBase], data_prep: pl.DataFrame):
    import polars as pl
    train_dataset, _ = dataset
    training_data = train_dataset.create_training_data()
    training_splits: list[TrainDataset] = []
    l = len(training_data)
    remaining = train_dataset

    s = (l // 10)
    for i in range(0, 9):
        split, remaining = remaining.split(s)
        training_splits.append(split)

    from futureish.gpr.gpytorch.backends.online_pep.gpytorch_online_pep import GPyTorchOnlinePEP
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer
    import gpytorch
    import numpy as np
    import torch
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    np.random.seed(0)

    def kernel_factory(batch_shape, x_dims, X, Y):
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=x_dims, batch_shape=batch_shape)
        kernel.lengthscale = torch.tensor(1.0)
        covar_module = gpytorch.kernels.ScaleKernel(kernel, batch_shape=batch_shape)
        covar_module.outputscale = torch.tensor(1.0)
        return covar_module

    model = Forecaster(GPyTorchTrainer(induce_size=20, kernel_factory=kernel_factory, basemodel_class=GPyTorchOnlinePEP))

    best_mse = None

    from futureish.datasets import TestDataset
    pred_dataset = TestDataset.create_with_config(data_prep, train_dataset.config)

    for split in training_splits:
        model.fit_step(split, config=GPyTorchFitConfig(iters=100))
        new_mse = model.calculate_mse(pred_dataset)["value_error"]
        if best_mse is not None and new_mse > best_mse * 1.05:
            raise AssertionError(f"MSE increased from {best_mse} to {new_mse}")
        if best_mse is None or new_mse < best_mse:
            best_mse = new_mse