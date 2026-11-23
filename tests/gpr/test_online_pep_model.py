from futureish.forecasting.forecaster import Forecaster
import pytest

from futureish.gpr.gpytorch.backends.online_pep.gpytorch_online_pep import GPyTorchOnlinePEP

pytest.importorskip("gpytorch")

import numpy as np
import torch
import gpytorch

from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchFitConfig

torch.set_default_dtype(torch.float64)


def _kernel_factory(ard_num_dims=3):
    def kernel_factory(batch_shape, x_dims, X, Y):
        base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
        return gpytorch.kernels.ScaleKernel(base, batch_shape=batch_shape)

    return kernel_factory


def _likelihood_factory(noise=0.1, noise_prior=None):
    def likelihood_factory(batch_shape, x_dims, X, Y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape, noise_prior=noise_prior)
        likelihood.noise = torch.tensor(noise, dtype=torch.float64)
        return likelihood

    return likelihood_factory


def _make_wrapper(basemodel_class, use_cuda=False, ard_num_dims=3):
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    return GPyTorchTrainer(
        induce_size=10,
        kernel_factory=_kernel_factory(ard_num_dims),
        likelihood=_likelihood_factory(0.1),
        basemodel_class=basemodel_class,
        use_cuda=use_cuda,
    )


def test_fit_step_succeeds_on_first_call_for_online_basemodel():
    """Regression test: fit_step used to check hasattr(self._basemodel, "next_online_model"),
    but self._basemodel is None until the first fit() creates it, so the check
    always failed on the very first call regardless of basemodel_class."""

    X = np.random.randn(30, 3)
    features = {"target": np.random.randn(30, 1)}

    wrapper = _make_wrapper(GPyTorchOnlinePEP)
    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))
    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))  # online update on 2nd call


def test_fit_time_limit_stops_before_iters_exhausted():
    """time_limit is an alternative stopping criterion to iters: a small time_limit
    with a huge iters should stop well short of running all iters."""
    import time
    from futureish.gpr.gpytorch.backends.gpytorch_sgp import GPyTorchSGP

    X = np.random.randn(30, 3)
    features = {"target": np.random.randn(30, 1)}

    wrapper = _make_wrapper(GPyTorchSGP)
    start = time.monotonic()
    wrapper.fit(X, features, config=GPyTorchFitConfig(iters=1_000_000, time_limit=0.2, tqdm=False))
    elapsed = time.monotonic() - start

    assert elapsed < 5, "fit should stop close to time_limit, not run all iters"
    assert 0 < len(wrapper.loss_hist["target"]) < 1_000_000


def test_fit_bounds_whole_multi_feature_fit_by_time_limit():
    """A time_limit passed to MultiOutputAssemblyModel.fit/fit_step bounds the
    whole call: every feature is trained jointly as one batched model, so unlike
    the old one-model-per-feature loop there's no per-feature time budget to
    split - the same time_limit governs the fit regardless of how many features
    it covers."""
    import time
    import polars as pl
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    n = 30
    df = pl.DataFrame({
        "time": list(range(n)),
        "t0": list(range(n)),
        "X": pl.Series(np.random.randn(n, 3)),
        "f1": pl.Series(np.random.randn(n, 1)),
        "f2": pl.Series(np.random.randn(n, 1)),
        "f3": pl.Series(np.random.randn(n, 1)),
    })

    builder = GPyTorchTrainer(induce_size=10, kernel_factory=_kernel_factory(3), likelihood=_likelihood_factory(0.1), basemodel_class=GPyTorchOnlinePEP)
    model = Forecaster(builder)

    time_limit = 0.6
    start = time.monotonic()
    model.model.fit(df.get_column("X").to_numpy(), {"f1": df.get_column("f1").to_numpy(), "f2": df.get_column("f2").to_numpy(), "f3": df.get_column("f3").to_numpy()}, config=GPyTorchFitConfig(iters=1_000_000, time_limit=time_limit, tqdm=False))
    elapsed = time.monotonic() - start

    assert elapsed < time_limit + 5, "fit time should be bounded by time_limit"
    assert model.model.feature_names == ["f1", "f2", "f3"]


def test_online_model_fits_data_and_produces_realistic_uncertainty():
    """End-to-end check that the online GP backend (GPyTorchOnlinePEP), trained
    incrementally through MultiOutputAssemblyModel.fit_step (as in real streaming
    use) as a batch of one feature, actually learns the target function and
    reports sane uncertainty: predictions should track a known function, the
    predictive std shouldn't be degenerate, most true values should fall within
    the predicted interval, and uncertainty should grow when extrapolating away
    from the training data - the hallmark of a working (not just
    numerically-not-crashing) GP fit."""
    import polars as pl
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    rng = np.random.default_rng(0)
    n_train = 120
    noise_std = 0.05

    X_train = rng.uniform(-3, 3, size=(n_train, 1))
    y_train = np.sin(X_train[:, 0]) + rng.normal(0, noise_std, size=n_train)

    df = pl.DataFrame({
        "time": list(range(n_train)),
        "t0": list(range(n_train)),
        "X": pl.Series(X_train),
        # Dataset.create_training_data() stores targets as a 1-element array
        # column (via pl.concat_arr), so to_numpy() yields shape (N, 1) - match
        # that here, since a plain (N,) column breaks SGPR_PEP's err() broadcast.
        "y": pl.Series(y_train.reshape(-1, 1)),
    })

    builder = GPyTorchTrainer(induce_size=20, kernel_factory=_kernel_factory(1), likelihood=_likelihood_factory(noise_std ** 2), basemodel_class=GPyTorchOnlinePEP)
    model = Forecaster(builder)

    # Stream the training data in through fit_step in chunks, exercising the
    # online-update path (next_online_model) rather than a single batch fit.
    for split in np.array_split(np.arange(n_train), 3):
        chunk = df[split.tolist()]
        model.model.fit_step(chunk.get_column("X").to_numpy(), {"y": chunk.get_column("y").to_numpy()}, config=GPyTorchFitConfig(iters=400, tqdm=False))

    x0_test = np.linspace(-3, 3, 60)
    y_test_true = np.sin(x0_test)
    test_df = pl.DataFrame({"t0": list(range(60)), "X": pl.Series(x0_test.reshape(-1, 1))})


    pred = model.model.predict(test_df.get_column("X").to_numpy())
    mean_pred, std_pred = pred["y"][0], pred["y"][1]

    assert np.all(np.isfinite(std_pred)) and np.all(std_pred > 0), "predictive std must be finite and positive"

    mae = np.mean(np.abs(mean_pred - y_test_true))
    corr = np.corrcoef(mean_pred, y_test_true)[0, 1]
    assert mae < 0.15, f"predicted mean should track sin(x) closely, got MAE={mae}"
    assert corr > 0.95, f"predicted mean should correlate strongly with the true function, got corr={corr}"

    coverage = np.mean(np.abs(y_test_true - mean_pred) <= 2 * std_pred)
    assert coverage >= 0.6, f"most true values should fall within mean +/- 2*std, got coverage={coverage}"

    x0_far = np.linspace(8, 10, 20)
    far_df = pl.DataFrame({"t0": list(range(20)), "X": pl.Series(x0_far.reshape(-1, 1))})
    std_far = model.model.predict(far_df.get_column("X").to_numpy())["y"][1]

    assert std_far.mean() > std_pred.mean() * 2, (
        "uncertainty should grow when extrapolating well outside the training range, "
        f"got interpolation std={std_pred.mean()}, extrapolation std={std_far.mean()}"
    )


def test_fit_step_rejects_non_online_basemodel_on_first_call():
    """A basemodel_class that doesn't support online updates (e.g. GPyTorchSGP)
    must be rejected on the very first fit_step call, not only once a second
    call would try to invoke next_online_model(). fit_step checks the class's
    supports_online_fitting flag before doing any work, so no model is ever
    constructed for an unsupported basemodel_class."""
    from futureish.gpr.gpytorch.backends.gpytorch_sgp import GPyTorchSGP

    X = np.random.randn(30, 3)
    features = {"target": np.random.randn(30, 1)}

    wrapper = _make_wrapper(GPyTorchSGP)
    with pytest.raises(NotImplementedError):
        wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))
    assert wrapper._basemodel is None, "no model should be created for a rejected fit_step call"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fit_step_online_updates_preserve_wrapper_on_cuda():
    """Regression test: GPyTorchOnlinePEP had no .cuda() of its own, so
    self._basemodel.cuda() delegated through __getattr__ to
    underlying_model.cuda(), which moves the underlying model correctly but
    returns *it* (not the wrapper). GPyTorchTrainer.fit() then reassigned
    self._basemodel to that unwrapped module, silently discarding the
    GPyTorchOnlinePEP wrapper and its next_online_model method, so the
    following fit_step call raised NotImplementedError."""
    import torch
    
    X1 = np.random.randn(30, 3)
    features1 = {"target": np.random.randn(30, 1)}
    X2 = np.random.randn(30, 3)
    features2 = {"target": np.random.randn(30, 1)}

    wrapper = _make_wrapper(GPyTorchOnlinePEP, use_cuda=torch.cuda.is_available())

    wrapper.fit_step(X1, features1, config=GPyTorchFitConfig(iters=2, tqdm=False))
    assert isinstance(wrapper._basemodel, GPyTorchOnlinePEP)

    wrapper.fit_step(X2, features2, config=GPyTorchFitConfig(iters=2, tqdm=False))  # used to raise NotImplementedError
    assert isinstance(wrapper._basemodel, GPyTorchOnlinePEP)

    mean, std = wrapper.predict(X2)["target"]
    assert mean.shape == (30,)


def test_builder_rejects_a_basemodel_class_without_batch_support():
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer
    from futureish.gpr.gpytorch.backends.gpytorch_model import ABCGPyTorchModel

    class _NotBatchable(ABCGPyTorchModel):
        supports_batch_fitting = False

        def forward(self, x) -> gpytorch.distributions.MultivariateNormal: ...
        def objective(self): ...

    with pytest.raises(TypeError):
        GPyTorchTrainer(induce_size=10, kernel_factory=_kernel_factory(1), basemodel_class=_NotBatchable)


@pytest.mark.parametrize("basemodel_class_name", ["GPyTorchOnlinePEP", "GPyTorchSGP", "GPyTorchVGP"])
def test_single_feature_fit_uses_a_batch_of_one(basemodel_class_name):
    """There's no separate unbatched code path any more: a single-feature fit
    goes through the exact same batched machinery as a multi-feature one, just
    with batch_shape=(1,) - this checks that holds for every batch-capable
    GPyTorch backend, not just the online PEP model."""
    import polars as pl
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer
    from futureish.gpr.gpytorch.backends.gpytorch_sgp import GPyTorchSGP
    from futureish.gpr.gpytorch.backends.gpytorch_vgp import GPyTorchVGP

    basemodel_class = {"GPyTorchOnlinePEP": GPyTorchOnlinePEP, "GPyTorchSGP": GPyTorchSGP, "GPyTorchVGP": GPyTorchVGP}[basemodel_class_name]

    n = 40
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(n, 1))
    y = np.sin(X[:, 0]).reshape(-1, 1) + rng.normal(0, 0.05, size=(n, 1))
    df = pl.DataFrame({
        "time": list(range(n)),
        "t0": list(range(n)),
        "X": pl.Series(X),
        "y": pl.Series(y),
    })

    builder = GPyTorchTrainer(induce_size=10, kernel_factory=_kernel_factory(1), likelihood=_likelihood_factory(0.05 ** 2), basemodel_class=basemodel_class)
    model = Forecaster(builder)
    model.model.fit(df.get_column("X").to_numpy(), {"y": df.get_column("y").to_numpy()}, config=GPyTorchFitConfig(iters=50, tqdm=False))

    assert model.model.batch_shape == torch.Size([1]) # pyright: ignore[reportAttributeAccessIssue]
    pred = model.model.predict(df.get_column("X").to_numpy())
    mean, std = pred["y"][0], pred["y"][1]
    assert mean.shape == (n,)
    assert np.all(np.isfinite(std)) and np.all(std > 0)


def test_batched_multi_output_assembly_model_fits_and_predicts():
    """MultiOutputAssemblyModel.fit_step, given a builder with a kernel
    factory, trains every feature of the assembly jointly as one batched
    model (see GPyTorchTrainer)."""
    import polars as pl
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    rng = np.random.default_rng(0)
    n_train = 90
    noise_std = 0.05

    X_train = rng.uniform(-3, 3, size=(n_train, 1))
    y_sin = np.sin(X_train[:, 0]) + rng.normal(0, noise_std, size=n_train)
    y_cos = np.cos(X_train[:, 0]) + rng.normal(0, noise_std, size=n_train)

    df = pl.DataFrame({
        "time": list(range(n_train)),
        "t0": list(range(n_train)),
        "X": pl.Series(X_train),
        "sin_feature": y_sin,
        "cos_feature": y_cos,
    })

    builder = GPyTorchTrainer(induce_size=20, kernel_factory=_kernel_factory(1), likelihood=_likelihood_factory(noise_std ** 2), basemodel_class=GPyTorchOnlinePEP)
    model = Forecaster(builder)

    for split in np.array_split(np.arange(n_train), 3):
        chunk = df[split.tolist()]
        model.model.fit_step(chunk.get_column("X").to_numpy(), {"sin_feature": chunk.get_column("sin_feature").to_numpy(), "cos_feature": chunk.get_column("cos_feature").to_numpy()}, config=GPyTorchFitConfig(iters=300, tqdm=False))

    assert isinstance(model.model, GPyTorchTrainer)
    assert model.model.batch_shape == torch.Size([2])
    assert set(model.model.feature_names) == {"sin_feature", "cos_feature"}

    x0_test = np.linspace(-3, 3, 50)
    test_df = pl.DataFrame({"t0": list(range(50)), "X": pl.Series(x0_test.reshape(-1, 1))})

    mean_df = model.model.predict_mean(test_df.get_column("X").to_numpy())
    mae_sin = np.mean(np.abs(mean_df["sin_feature"] - np.sin(x0_test)))
    mae_cos = np.mean(np.abs(mean_df["cos_feature"] - np.cos(x0_test)))
    assert mae_sin < 0.15, f"batched sin_feature should track sin(x), got MAE={mae_sin}"
    assert mae_cos < 0.15, f"batched cos_feature should track cos(x), got MAE={mae_cos}"

    predictions = model.model.predict(test_df.get_column("X").to_numpy())
    for name in ("sin_feature", "cos_feature"):
        mean, std = predictions[name][0], predictions[name][1]
        assert np.all(np.isfinite(std)) and np.all(std > 0)

    # every feature was jointly optimized, but each keeps its own loss trace
    assert len(model.model.loss_hist["sin_feature"]) == len(model.model.loss_hist["cos_feature"])
    assert model.model.loss_hist["sin_feature"] != model.model.loss_hist["cos_feature"]


def test_batched_features_learn_independent_likelihood_noise():
    """Regression test: pep_compute_base used to construct its GaussianLikelihood
    with no batch_shape at all, so every jointly-trained feature shared a single
    noise parameter (and therefore a summed gradient) instead of each feature
    learning its own. Two features with very different noise levels should end
    up with correspondingly different learned noise, not converge to a shared
    compromise value."""
    import polars as pl
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    rng = np.random.default_rng(0)
    n = 80
    X = rng.uniform(-3, 3, size=(n, 1))
    y_quiet = np.sin(X[:, 0]) + rng.normal(0, 0.01, size=n)
    y_noisy = np.sin(X[:, 0]) + rng.normal(0, 1.0, size=n)

    df = pl.DataFrame({
        "time": list(range(n)),
        "t0": list(range(n)),
        "X": pl.Series(X),
        "quiet": y_quiet,
        "noisy": y_noisy,
    })

    builder = GPyTorchTrainer(induce_size=15, kernel_factory=_kernel_factory(1), likelihood=_likelihood_factory(0.1), basemodel_class=GPyTorchOnlinePEP)
    model = Forecaster(builder)
    model.model.fit(df.get_column("X").to_numpy(), {"quiet": df.get_column("quiet").to_numpy(), "noisy": df.get_column("noisy").to_numpy()}, config=GPyTorchFitConfig(iters=300, lr=0.05, tqdm=False))

    noise = model.model._basemodel.likelihood.noise.detach().flatten().tolist() # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    feature_names = model.model.feature_names
    noise_by_feature = dict(zip(feature_names, noise))

    assert noise_by_feature["noisy"] > noise_by_feature["quiet"] * 5, (
        f"the noisy feature should learn a much larger noise level than the quiet one, got {noise_by_feature}"
    )


def test_kernel_and_likelihood_priors_contribute_to_the_objective():
    """PEP's bound is hand-rolled rather than built through gpytorch.mlls, so
    priors registered on the kernel (lengthscale_prior=... in the kernel
    factory) or the likelihood (built with noise_prior=... and handed to
    `likelihood=` directly) don't affect the loss unless objective()
    explicitly adds their log_prob on top of _maximum_log_likelihood_objective().
    _maximum_log_likelihood_objective() itself must stay the plain PEP bound
    (other tests pin it against golden values), so only objective() should
    change. The prior's log_prob is scaled by N (the batch size) since mll
    sums a per-point term over all N observations while a prior contributes
    only one N-independent term - left unscaled, the prior would be swamped
    by the likelihood for any non-trivial N."""
    from futureish.gpr.gpytorch.backends.online_pep.sgpr_pep import SGPR_PEP

    rng = np.random.default_rng(0)
    n = 20
    X = torch.tensor(rng.uniform(-3, 3, size=(n, 1)))
    Y = (torch.sin(X[:, 0]) + torch.tensor(rng.normal(0, 0.1, size=n))).unsqueeze(-1)
    Z = X[:10].clone()

    lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
    noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
    kernel = gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)
    kernel.lengthscale = 0.7 # pyright: ignore[reportAttributeAccessIssue]

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)
    likelihood.noise = torch.tensor(0.1, dtype=torch.float64)

    model = SGPR_PEP(X, Y, kernel=kernel, likelihood=likelihood, z=Z)

    mll = model._maximum_log_likelihood_objective()
    prior_log_prob = sum(
        ((prior.log_prob(closure(module)) * model.N).sum() for _, module, prior, closure, _ in model.named_priors()),
        start=torch.tensor(0.0, dtype=mll.dtype),
    )

    assert prior_log_prob.item() != 0.0, "test priors should contribute a non-zero log_prob"
    assert pytest.approx((-(mll + prior_log_prob)).item()) == model.objective().item()

    plain_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    plain_likelihood.noise = torch.tensor(0.1, dtype=torch.float64)
    plain_model = SGPR_PEP(X, Y, kernel=gpytorch.kernels.RBFKernel(), likelihood=plain_likelihood, z=Z)
    plain_model.covar_module.lengthscale = kernel.lengthscale.detach().clone() # pyright: ignore[reportAttributeAccessIssue]
    assert pytest.approx(-plain_model._maximum_log_likelihood_objective().item()) == plain_model.objective().item(), (
        "with no priors registered, objective() must reduce to -_maximum_log_likelihood_objective()"
    )


def test_online_pep_wrapper_threads_a_custom_likelihood_through_to_the_basemodel():
    """GPyTorchOnlinePEP/GPyTorchTrainer accept a `likelihood` factory
    (same dual-mode contract as `kernel`/`kernel_factory`) rather than a bare
    noise value, so the caller can hand over a fully-configured Likelihood -
    e.g. one built with noise_prior=... - and it must survive the online
    update to OSGPR_PEP (which deep-copies the likelihood)."""
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
    wrapper = GPyTorchTrainer(
        induce_size=10,
        kernel_factory=_kernel_factory(3),
        likelihood=_likelihood_factory(0.1, noise_prior=noise_prior),
        basemodel_class=GPyTorchOnlinePEP,
    )

    X = np.random.randn(30, 3)
    features = {"target": np.random.randn(30, 1)}
    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))

    assert wrapper._basemodel is not None
    assert wrapper._basemodel.likelihood.noise_covar.noise_prior is noise_prior

    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))  # triggers next_online_model

    assert wrapper._basemodel.likelihood.noise_covar.noise_prior is not None, (
        "noise_prior should survive the deepcopy into OSGPR_PEP on the online update"
    )


def test_online_pep_recenters_a_normal_noise_prior_between_steps_without_changing_its_scale():
    """A NormalPrior noise_prior should track this step's converged noise value into
    the next one's prior - mirroring how mu_old/Su_old already carry the GP's
    posterior over inducing outputs forward. Only the mean moves: the scale is left
    exactly as the caller originally configured it, so the prior's strength doesn't
    drift on its own between steps."""
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    original_scale = torch.tensor(1.0)
    noise_prior = gpytorch.priors.NormalPrior(loc=torch.tensor(10.0), scale=original_scale)
    wrapper = GPyTorchTrainer(
        induce_size=10,
        kernel_factory=_kernel_factory(3),
        likelihood=_likelihood_factory(10.0, noise_prior=noise_prior),
        basemodel_class=GPyTorchOnlinePEP,
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    features = {"target": rng.normal(scale=5.0, size=200)}

    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=20, tqdm=False))
    noise_after_step1 = wrapper._basemodel.likelihood.noise_covar.noise.detach().clone() # type: ignore

    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=20, tqdm=False))  # triggers next_online_model

    assert wrapper._basemodel is not None
    updated_prior = wrapper._basemodel.likelihood.noise_covar.noise_prior
    assert isinstance(updated_prior, gpytorch.priors.NormalPrior)
    assert updated_prior is not noise_prior, "the prior object itself should be replaced, not mutated in place"
    assert torch.allclose(updated_prior.loc, noise_after_step1), (
        "the new prior should be recentered at the noise value the previous step converged to"
    )
    assert bool((updated_prior.scale == original_scale).all()), "the scale must be left untouched"


def test_online_pep_recenters_a_gamma_noise_prior_between_steps_without_changing_its_concentration():
    """A GammaPrior noise_prior should track this step's converged noise value into
    the next one's prior too, exactly like a NormalPrior does - just moved via the
    Gamma's mode instead of a Normal's loc. Only the rate moves to relocate the mode:
    concentration (the prior's shape/sharpness) is left exactly as the caller
    originally configured it, so the prior stays a GammaPrior rather than being
    coerced into some other family."""
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    noise_prior = gpytorch.priors.GammaPrior(3.0, 0.05)
    wrapper = GPyTorchTrainer(
        induce_size=10,
        kernel_factory=_kernel_factory(3),
        likelihood=_likelihood_factory(0.1, noise_prior=noise_prior),
        basemodel_class=GPyTorchOnlinePEP,
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    features = {"target": rng.normal(scale=5.0, size=200)}

    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=20, tqdm=False))
    assert wrapper._basemodel is not None
    noise_after_step1 = wrapper._basemodel.likelihood.noise_covar.noise.detach().clone() # pyright: ignore[reportCallIssue]

    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=20, tqdm=False))  # triggers next_online_model

    updated_prior = wrapper._basemodel.likelihood.noise_covar.noise_prior
    assert isinstance(updated_prior, gpytorch.priors.GammaPrior)
    assert updated_prior is not noise_prior, "the prior object itself should be replaced, not mutated in place"
    assert torch.allclose(updated_prior.concentration, noise_prior.concentration), (
        "concentration must be left untouched"
    )
    mode = (updated_prior.concentration - 1) / updated_prior.rate
    assert torch.allclose(mode, noise_after_step1), (
        "the new prior's mode should be recentered at the noise value the previous step converged to"
    )


def test_online_pep_leaves_a_gamma_noise_prior_with_no_interior_mode_untouched():
    """A GammaPrior whose concentration is too low for an interior mode to exist
    should survive the online update exactly as-is rather than being recentered
    onto a mode that doesn't exist, or coerced into some other family."""
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    noise_prior = gpytorch.priors.GammaPrior(1.0, 0.05)  # concentration <= 1: no mode to recenter onto
    wrapper = GPyTorchTrainer(
        induce_size=10,
        kernel_factory=_kernel_factory(3),
        likelihood=_likelihood_factory(0.1, noise_prior=noise_prior),
        basemodel_class=GPyTorchOnlinePEP,
    )

    X = np.random.randn(30, 3)
    features = {"target": np.random.randn(30, 1)}
    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))
    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))  # triggers next_online_model

    assert wrapper._basemodel is not None
    updated_prior = wrapper._basemodel.likelihood.noise_covar.noise_prior
    assert isinstance(updated_prior, gpytorch.priors.GammaPrior)
    assert torch.allclose(updated_prior.concentration, noise_prior.concentration)
    assert torch.allclose(updated_prior.rate, noise_prior.rate)


def test_online_pep_raises_for_a_noise_prior_family_it_cannot_recenter():
    """A noise_prior family with no case in _recentered_noise_prior (e.g. a
    LogNormalPrior) should raise NotImplementedError rather than silently being
    left un-recentered forever."""
    from futureish.gpr.gpytorch.gpytorch_model_trainer import GPyTorchTrainer

    noise_prior = gpytorch.priors.LogNormalPrior(loc=torch.tensor(-2.0), scale=torch.tensor(0.5))
    wrapper = GPyTorchTrainer(
        induce_size=10,
        kernel_factory=_kernel_factory(3),
        likelihood=_likelihood_factory(0.1, noise_prior=noise_prior),
        basemodel_class=GPyTorchOnlinePEP,
    )

    X = np.random.randn(30, 3)
    features = {"target": np.random.randn(30, 1)}
    wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))
    with pytest.raises(NotImplementedError):
        wrapper.fit_step(X, features, config=GPyTorchFitConfig(iters=2, tqdm=False))  # triggers next_online_model

    assert wrapper._basemodel is not None
    assert wrapper._basemodel.likelihood.noise_covar.noise_prior is noise_prior
