import pytest
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from futureish.datasets import OnlineDataset

def test_catboost_import():
    import catboost
    assert catboost.__version__ is not None

def test_catboost_model():
    from futureish.catboost.model import CatBoostTimeseriesModel
    model = CatBoostTimeseriesModel()
    assert model is not None

@pytest.fixture(params=[1, 2])
def time_series_dataset(request: pytest.FixtureRequest) -> "OnlineDataset":
    n_targets = request.param
    import numpy as np
    from futureish.datapoints import RecordedDataPoint
    from futureish.features import RecordedInputFeature, TargetFeature
    from futureish.datasets import OnlineDataset
    import polars as pl
    from datetime import datetime, timedelta

    data = pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 12, 31),
            interval="1d",
            eager=True,
        )
    }).with_columns(
        pl.int_range(0, pl.len()).alias("feature"),
        pl.int_range(0, pl.len()).alias("value"),
    )
    datapoint_feature = RecordedDataPoint(name="feature", step_size="1d", known_future=True)
    datapoint_value = RecordedDataPoint(name="value", step_size="1d")

    recorded_input_feature = RecordedInputFeature(datapoint_feature, feature_name="feature_feature")

    targets = []
    for i in range(n_targets):
        targets.append(TargetFeature(datapoint_value, feature_name=f"value_target_{i}"))
        
    dataset = OnlineDataset(
        data,
        input_features=[recorded_input_feature],
        target_features=targets,
        size_of_first_online_window="5d",
    )
    return dataset

@pytest.mark.parametrize("use_multioutput", [False, True])
def test_catboost_model_fit_predict(time_series_dataset: "OnlineDataset", use_multioutput: bool):
    from futureish.catboost.model import CatBoostTimeseriesModel, CatboostFitConfig
    from futureish.forecasting import Forecaster

    train, predict = time_series_dataset.split_training_and_test_data(0.5)

    model = Forecaster(CatBoostTimeseriesModel(iterations=100, allow_writing_files=False, use_multioutput=use_multioutput))

    model.fit(train, config=CatboostFitConfig(verbose=False))

    traj, ground = predict.create_prediction_trajectory(pred_len="7d")
    model.predict_autoregressive(traj)
    model.predict_mean(predict.create_all_single_inputs())
    model.calculate_mse(predict)

@pytest.mark.parametrize("samples", [1, 10])
def test_catboost_model_fit_predict_uncertainty(time_series_dataset: "OnlineDataset", samples: int):
    from futureish.catboost.model import CatBoostTimeseriesModel, CatboostFitConfig
    from futureish.forecasting import Forecaster

    train, predict = time_series_dataset.split_training_and_test_data(0.5)

    model = Forecaster(CatBoostTimeseriesModel(iterations=100, allow_writing_files=False, uncertainty_prediction=True))

    model.fit(train, config=CatboostFitConfig(verbose=False))

    traj, ground = predict.create_prediction_trajectory(pred_len="7d")
    model.predict_autoregressive(traj, samples=samples)
    model.predict_mean(predict.create_all_single_inputs())
    model.predict(predict.create_all_single_inputs())
    model.calculate_mse(predict)