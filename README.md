# futureish

`futureish` is a Python time-series forecasting library built to create probabilistic time series models with online update of models.
Its main feature is a online gaussian process model, but also features a Catboost model and allows easy implementation of other black-box models.
It features a powerfull interface and functions for defining inputs, targets and datasets

- Recorded, sparse, modular, calendar, cyclical, and bucketed input features
- Target features and configurable data preparation
- Z-score, Difference, and chained scalers
- Online dataset workflows
- GPyTorch Gaussian-process backends, including sparse and online PEP models
- CatBoost backend

## Installation

```bash
pip install futureish
```
or
```bash
uv add futureish
```

## Quick start

The following example creates a one-minute dataset, derives a lagged input
feature, and prepares training data:

```python
from futureish.datapoints import RecordedDataPoint
from futureish.features import RecordedInputFeature, BusinessDayFeature, SinCosTimeFeature
from futureish.scaling import DiffScaler, ZScoreScaler, ChainedScaler
from futureish.features import TargetFeature
from futureish.datasets import Dataset
from futureish.gpr import GPyTorchOnlinePEP, GPyTorchFitConfig
from futureish.forecasting import Forecaster
import torch

pas_dp = RecordedDataPoint(name="pas", step_size="30m", known_future=False) 

input_features = [
    RecordedInputFeature(pas_dp, feature_name="pas_lags", input_length="2h", scaler_factory=ChainedScaler.from_scalers(DiffScaler, ZScoreScaler)),
]
target_features = [
    TargetFeature(pas_dp, feature_name="pas_feature", scaler_factory=ChainedScaler.from_scalers(DiffScaler, ZScoreScaler))
]

dataset = Dataset(
    raw_data=data,
    input_features=input_features,
    target_features=target_features,
)

train_dataset, test_dataset = dataset.split_training_and_test_data(0.8)

model = Forecaster(GPyTorchOnlinePEP.wrapped(100, use_cuda=torch.cuda.is_available()))
model.fit(train_dataset, config=GPyTorchFitConfig(time_limit=120)) # train for 120 seconds
traj_data, ground_truth = test_dataset.create_prediction_trajectory(t0_offset="6h", pred_len="2d") 
traj = model.predict_autoregressive(traj_data)
traj_sampled = model.predict_autoregressive(traj_data, samples=100)
traj_quantiles = (
    traj_sampled
    .group_by("time")
    .agg([
        pl.col("pas").quantile(0.1).alias("pas_10th_percentile"),
        pl.col("pas").quantile(0.5).alias("pas_median"),
        pl.col("pas").quantile(0.9).alias("pas_90th_percentile")
    ])
)
```
For full example see: [Example](examples/example.ipynb)


Raw data must contain a `time` column and the columns referenced by the
configured datapoints. Feature names must be unique, and target datapoints
define the forecasting step size.

## Components:

### Datapoints

`RecordedDataPoint` describes a value recorded in the raw data. It defines the
column name, recording step size, and whether future values are known. A
datapoint is the common source used by input and target features.
```python
from futureish.datapoints import RecordedDataPoint
```

### Input Features
```python 
from futureish.features import *
```
#### RecordedInputFeature
A basic input feature that includes the current value of the datapoint and optionally lags as defined in `input_length` and $t_1$ if `uses_future_data=True`.
`input_lenght` defines the lags, `None` results in only $t_0$ and a value defines how much in the past is included.

#### SparseInputFeature
Some functionality, but only uses every nth lag as defined in reduction factor. For example `reduction_factor=2` uses lags $t_0, t_{-2}, ...$

#### ModularInputFeature
Allows to create a custom function for creating features. The function gets a `pl.DataFrame` or `pl.LazyFrame`, the column name for the time, the name for the feature and the groups (which include t0)

#### BucketInputFeature
Creates buckets for the lags, that each can be modified or aggregated by a custom function defining the `pl.Expr`

#### SinCosTimeFeature
Cyclic time features for different time variable encoded with sin and cos

#### BusinessDayFeature
Creates a indicator if the current day is a business day or not. Can also include indications for previos days according the `day_history`

#### InputFeatureBase
Base class to create custom features. At minimum a `_create_features` needs to be implemented. For more control `create_features` can be implemented.

### Target Features

`TargetFeature` converts a recorded datapoint into the values the model learns
to predict and maps it back to the raw data format. Target features define the forecast outputs, can apply scaling and
optional bounds, and determine the forecasting step size used by the dataset.

### Scalers

Scalers transform feature values before model fitting and invert predictions
back to the original scale. Built-in scalers include `ZScoreScaler`, `DiffScaler` and `ChainedScaler` composes multiple scalers
when several transformations are needed.

#### ZScoreScaler

```python 
(x - x.mean()) / x.std()
```

#### DiffScaler

$x = t_{-1} - t_0$

#### Custom Scalers

Custom scalers can be implemented as a subclass from `ScalerBase` (interface) or `DataScaler` (Simplified if the scaling can be implemented as a simple `pl.Expr`)

### Dataset

`Dataset` combines raw time-series data with input and target features. It
validates and prepares the feature columns, fits scaling configuration on
training data, and can produce `TrainDataset` and `TestDataset` instances.
Specialized online, prediction, and forecasting datasets support incremental
updates and autoregressive prediction workflows.
The datasets are split to ensure seperation between training, test and prediction data.

The basic dataset can be split into train and test datasets with `split_training_and_test_data` or converted entirely to a training dataset with `to_train_dataset`
Also has a `iter` and `random_subset` method.

#### TrainDataset
Can be passed on to the fit method to train the model.

##### OnlineDataset
The same as the basic dataset, but specifiying the size of the first training step, which limits the amount of data that is available for fitting the scalers.

#### OnlineTrainingDataset
The training dataset for the `OnlineDataset`

#### TestDataset
Used to test the model performance and make predictions. 
- `create_prediction_trajectory` creates a single prediction trajectory
- `create_all_trajectories` create all possible trajectories based on the data
- `create_all_single_inputs` creates all inputs to predict $t_{+1}$

### Forecaster

`Forecaster` holds the model and the configuration for the dataset. It provides fitting,
incremental fitting, point or distributional predictions, autoregressive
multi-step forecasts, samples, and forecast error metrics.

### Models

Models implement the common model interface used by `Forecaster`, allowing
different backends to be exchanged without changing the data pipeline. The
project currently provides GPyTorch Gaussian-process models, including sparse
and online PEP variants, and a CatBoost time-series model.

#### GPR Models
All models are implemented in Gpytorch and provide native uncertainty estimation
- `GPyTorchOnlinePEP` Online Trainable GPR model
- `GPyTorchSGP` Comparison model
- `GPyTorchVGP` Comparison model

All three models need to be wrapped to be used with the `Forecaster` class.
```python
Forecaster(GPyTorchOnlinePEP.wrapped(...))
```
Each can be customized with the number of inducing points and a custom Kernel based on Gpytorch Kernels.

#### Catboost
A wrapper for catboost, that can also provide uncertainty estimation.

#### Custom models
Custom models can be created by subclassing `ABCModel` and implementing `_fit` and `_predict_mean` and optionally `_predict` and `_predict_sample`.

## Status

This project is under active development. Public APIs and model behavior may
change as the library evolves.

But feel free to open issues if you encounter them or have questions.

**Copyright © [2026] Technical University of Denmark**