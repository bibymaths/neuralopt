"""Track A: compiled preconditioned engine — parity with the Python loop, batching, caching."""

from __future__ import annotations

import logging
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import Problem, minimize
from neuralopt.compiled import preconditioned_solve_batch
from neuralopt.model import ModelConfig, init_mlp_model, init_model
from neuralopt.training import load_checkpoint

HISTORY = ModelConfig(
    width=16, heads=4, rank=2, layers=1, scale_aware=True, geometry_bound=10.0,
    lowrank_scale=0.5, feature_set="history-v1", history_length=4,
)
LEGACY = replace(HISTORY, feature_set="legacy", history_length=0)


def _rotated(dtype, d=5, seed=0, budget=60):
    key = jax.random.PRNGKey(seed)
    q, _ = jnp.linalg.qr(jax.random.normal(key, (d, d), dtype=dtype))
    matrix = q @ jnp.diag(jnp.geomspace(1.0, 100.0, d).astype(dtype)) @ q.T
    lo = jnp.linspace(-3.0, -1.0, d, dtype=dtype)
    hi = jnp.linspace(1.0, 6.0, d, dtype=dtype)
    center = 0.5 * (lo + hi) + 0.3

    def objective(x):
        return 0.5 * (x - center) @ matrix @ (x - center)

    return Problem(objective, (lo, hi), d, max_evaluations=budget, dtype=dtype)


def _rastrigin(dtype, d=4, budget=60):
    def objective(x):
        z = x - 0.3
        return 10.0 * d + jnp.sum(z**2 - 10.0 * jnp.cos(2 * jnp.pi * z))

    return Problem(objective, (-5.0, 5.0), d, x0=np.full(d, 1.7), max_evaluations=budget,
                   dtype=dtype)


ARMS = [
    ("history-transformer", HISTORY, "transformer", "learned"),
    ("history-mlp", HISTORY, "mlp", "learned"),
    ("legacy-transformer", LEGACY, "transformer", "learned"),
    ("fixed", LEGACY, "transformer", "fixed"),
    ("identity", LEGACY, "transformer", "identity"),
]


def _run(problem, engine, params, config, model_kind, geometry, record=True):
    return minimize(
        problem,
        params=params,
        config=config,
        model_kind=model_kind,
        candidate_method="preconditioned",
        hamiltonian_metric=geometry,
        fixed_metric_diag=np.full(problem.dimension, 1.5),
        preconditioned_step=0.2,
        record_diagnostics=record,
        preconditioned_engine=engine,
    )


@pytest.mark.parametrize("make_problem", [_rotated, _rastrigin], ids=["rotated", "rastrigin"])
@pytest.mark.parametrize(("name", "config", "model_kind", "geometry"), ARMS,
                         ids=[a[0] for a in ARMS])
def test_float64_compiled_engine_matches_python_loop(make_problem, name, config, model_kind,
                                                     geometry):
    with jax.enable_x64(True):
        problem = make_problem(jnp.float64)
        init = init_model if model_kind == "transformer" else init_mlp_model
        params = init(jax.random.PRNGKey(1), config)
        python = _run(problem, "python", params, config, model_kind, geometry)
        compiled = _run(problem, "compiled", params, config, model_kind, geometry)
    dp, dc = python.diagnostics, compiled.diagnostics
    for key in ("objective_evaluations", "gradient_evaluations", "accepted_steps",
                "rejected_steps", "hvp_evaluations"):
        assert dp[key] == dc[key], key
    assert python.status == compiled.status and python.evaluations == compiled.evaluations
    np.testing.assert_allclose(compiled.trajectory, python.trajectory, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(compiled.x, python.x, rtol=1e-10, atol=1e-12)
    assert [r["accepted"] for r in dc["records"]] == [r["accepted"] for r in dp["records"]]
    np.testing.assert_allclose([r["step_length"] for r in dc["records"]],
                               [r["step_length"] for r in dp["records"]], rtol=1e-10)
    hp, hc = dp["history"], dc["history"]
    assert hp["valid"] == hc["valid"] and hp["rejections"] == hc["rejections"]
    for field in ("u", "g", "f"):
        np.testing.assert_allclose(hc[field], hp[field], rtol=1e-10, atol=1e-12)
    assert dc["engine"] == "compiled" and dp["engine"] == "python"


def test_float32_compiled_engine_is_statistically_equivalent():
    """Hard acceptance can flip on float32 ties; outcomes must agree in distribution."""
    params, config = load_checkpoint("artifacts/trajectory-v0.6.0-tf-h8-seed0.npz")
    differences, identical = [], 0
    for seed in range(12):
        problem = _rotated(jnp.float32, d=6, seed=seed, budget=100)
        python = _run(problem, "python", params, config, "transformer", "learned", False)
        compiled = _run(problem, "compiled", params, config, "transformer", "learned", False)
        f0 = python.trajectory[0]
        differences.append(np.log10(compiled.fun / f0 + 1e-10) - np.log10(python.fun / f0 + 1e-10))
        identical += int(python.diagnostics["accepted_steps"]
                         == compiled.diagnostics["accepted_steps"])
        assert compiled.evaluations <= 100 and python.evaluations <= 100
    assert abs(float(np.mean(differences))) < 0.1
    assert float(np.median(np.abs(differences))) < 0.05
    assert identical >= 6


def test_compiled_engine_reports_errors_and_terminal_states():
    params, config = init_model(jax.random.PRNGKey(0), HISTORY), HISTORY
    bad = {**params, "diag": params["diag"] * jnp.nan}
    with pytest.raises(FloatingPointError, match="geometry"):
        _run(_rotated(jnp.float32), "compiled", bad, config, "transformer", "learned")
    nan_start = Problem(lambda x: jnp.sum(x) * jnp.nan, (-1.0, 1.0), 2)
    with pytest.raises(FloatingPointError, match="initial"):
        _run(nan_start, "compiled", params, config, "transformer", "learned")
    flat = Problem(lambda x: jnp.sum(x * 0.0), (-1.0, 1.0), 2)
    assert _run(flat, "compiled", params, config, "transformer", "learned").status == "stationary"
    # Gradient pushes against the lower bound: every candidate is the start point.
    stuck = Problem(lambda x: jnp.sum(x), (-1.0, 1.0), 2, x0=np.array([-1.0, -1.0]),
                    max_evaluations=20)
    result = _run(stuck, "compiled", params, config, "transformer", "learned")
    assert result.diagnostics["accepted_steps"] == 0
    assert result.diagnostics["rejected_steps"] == 17 and result.evaluations == 19


def test_compiled_engine_requires_traceable_objective():
    problem = Problem(lambda x: float(np.sum(np.asarray(x) ** 2)), (-1.0, 1.0), 2,
                      gradient=lambda x: 2 * jnp.asarray(x))
    with pytest.raises(ValueError, match="traceable"):
        _run(problem, "compiled", None, LEGACY, "transformer", "fixed")


def test_repeated_compiled_solves_do_not_recompile():
    problem = _rotated(jnp.float32)
    params = init_model(jax.random.PRNGKey(0), HISTORY)
    _run(problem, "compiled", params, HISTORY, "transformer", "learned")
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture()
    logging.getLogger("jax").addHandler(handler)
    try:
        with jax.log_compiles(True):
            other = {k: v * 1.01 for k, v in params.items()}  # new weights, same shapes
            _run(problem, "compiled", other, HISTORY, "transformer", "learned")
    finally:
        logging.getLogger("jax").removeHandler(handler)
    assert not [m for m in messages if "Compiling" in m]


def _parametric_rotated(x, theta):
    center, matrix = theta
    return 0.5 * (x - center) @ matrix @ (x - center)


def test_batched_solve_matches_single_solves_and_charges_gradients_on_acceptance():
    d, tasks, budget = 4, 5, 50
    with jax.enable_x64(True):
        keys = jax.random.split(jax.random.PRNGKey(3), tasks)
        qs = jax.vmap(lambda k: jnp.linalg.qr(jax.random.normal(k, (d, d)))[0])(keys)
        spectrum = jnp.geomspace(1.0, 100.0, d)
        matrices = jnp.einsum("bij,j,bkj->bik", qs, spectrum, qs)
        centers = jax.vmap(lambda k: jax.random.uniform(k, (d,), minval=-0.5, maxval=0.5))(keys)
        lo, hi = -jnp.ones((tasks, d)) * 2.0, jnp.ones((tasks, d)) * 2.0
        x0 = jnp.zeros((tasks, d)) + 1.2
        params = init_model(jax.random.PRNGKey(2), HISTORY)
        batch = preconditioned_solve_batch(
            _parametric_rotated, (centers, matrices), x0, (lo, hi), budget,
            params=params, config=HISTORY,
        )
        for i in range(tasks):
            center, matrix = centers[i], matrices[i]

            def objective(x, center=center, matrix=matrix):
                return _parametric_rotated(x, (center, matrix))

            single = minimize(
                Problem(objective, (lo[i], hi[i]), d, x0=np.asarray(x0[i]), dtype=jnp.float64),
                max_evaluations=budget, params=params, config=HISTORY,
                candidate_method="preconditioned", preconditioned_engine="compiled",
            )
            d_single = single.diagnostics
            assert int(batch["objective_evaluations"][i]) == d_single["objective_evaluations"]
            assert int(batch["gradient_evaluations"][i]) == d_single["gradient_evaluations"]
            np.testing.assert_allclose(float(batch["fun"][i]), single.fun, rtol=1e-10)
    # Under vmap lax.cond evaluates both branches, but only acceptances are charged.
    accepted = np.asarray(batch["accepted_steps"])
    np.testing.assert_array_equal(np.asarray(batch["gradient_evaluations"]), 1 + accepted)
    iterations = np.asarray(batch["objective_evaluations"]) - 1
    assert np.any(accepted < iterations)  # some rejections happened in the batch
    assert np.all(np.asarray(batch["objective_evaluations"])
                  + np.asarray(batch["gradient_evaluations"]) <= budget)
