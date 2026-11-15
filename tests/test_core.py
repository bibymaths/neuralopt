from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import ModelConfig, Problem, Variable, init_model, minimize, verify_certificate
from neuralopt.benchmark import run_baseline
from neuralopt.certification import certify
from neuralopt.features import (
    denormalize_coordinates,
    normalize_coordinates,
    parameter_features,
)
from neuralopt.model import apply_mlp_model, apply_model, init_mlp_model, precondition


def test_problem_validation():
    with pytest.raises(ValueError):
        Problem(lambda x: jnp.sum(x), (1, -1), 2)
    with pytest.raises(ValueError):
        Problem(lambda x: jnp.sum(x), (-1, 1), 2, x0=jnp.zeros(3))


def test_model_shapes_and_positive_preconditioner():
    c = ModelConfig(width=16, heads=4, rank=2, layers=1)
    p = init_model(jax.random.PRNGKey(0), c)
    f = parameter_features(jnp.zeros(7), jnp.ones(7), -jnp.ones(7), jnp.ones(7), jnp.zeros(7))
    out = jax.jit(lambda z: apply_model(p, z, 1.0, c))(f)
    assert out["direction"].shape == (7,) and out["lowrank"].shape == (7, 2)
    assert np.all(np.asarray(out["diag"]) > 0)
    v = jnp.arange(7.0)
    pv = precondition(v, out["diag"], out["lowrank"])
    assert float(v @ pv) > 0
    mlp = init_mlp_model(jax.random.PRNGKey(1), c)
    mlp_out = apply_mlp_model(mlp, f, 1.0, c)
    assert mlp_out["direction"].shape == (7,)
    assert sum(x.size for x in jax.tree_util.tree_leaves(mlp)) < sum(
        x.size for x in jax.tree_util.tree_leaves(p)
    )


def test_deterministic_learned_minimize_and_bounds():
    problem = Problem(lambda x: jnp.sum((x - 0.3) ** 2), (-1.0, 1.0), 3, max_evaluations=12)
    a = minimize(problem, seed=4)
    b = minimize(problem, seed=4)
    np.testing.assert_allclose(a.x, b.x)
    assert a.fun == b.fun
    assert np.all(np.asarray(a.x) >= -1) and np.all(np.asarray(a.x) <= 1)
    assert len(a.trajectory) == 12 and np.all(np.diff(a.trajectory) <= 0)


def test_coordinate_roundtrip_and_diagnostics():
    lo, hi = jnp.array([-2.0, 1.0]), jnp.array([4.0, 9.0])
    x = jnp.array([0.5, 3.0])
    np.testing.assert_allclose(denormalize_coordinates(normalize_coordinates(x, lo, hi), lo, hi), x)
    problem = Problem(lambda z: jnp.sum((z - 0.2) ** 2), (lo, hi), 2, x0=x)
    result = minimize(problem, max_evaluations=6, record_diagnostics=True)
    records = result.diagnostics["records"]
    assert len(records) == 5
    assert all(np.isfinite(r["raw_alignment"]) for r in records)
    assert all(r["preconditioner_min_eigenvalue"] > 0 for r in records)
    assert 0.0 <= result.diagnostics["acceptance_rate"] <= 1.0


def test_certified_quadratic_and_replay():
    x = Variable()
    expr = (x[0] - 0.25) ** 2 + (x[1] + 0.5) ** 2
    problem = Problem(
        lambda z: jnp.sum((z - jnp.array([0.25, -0.5])) ** 2),
        (-1.0, 1.0),
        2,
        certified_expression=expr,
        identifier="q",
    )
    result = minimize(problem, mode="certified", absolute_gap=2e-2, max_evaluations=2000)
    assert result.status == "certified_gap"
    assert result.lower_bound <= 0 <= result.upper_bound
    assert verify_certificate(result.certificate)


def test_certified_incumbent_preserves_binary64_precision():
    x = Variable()
    expr = (x[0] - 0.1) ** 2
    problem = Problem(
        lambda z: (z[0] - 0.1) ** 2,
        (0.0, 1.0),
        1,
        certified_expression=expr,
    )
    result = minimize(problem, mode="certified", absolute_gap=1e-8, max_evaluations=1000)
    assert result.x.dtype == np.float64
    point_upper = float(expr.interval(result.x, result.x).hi)
    assert point_upper <= result.upper_bound


def test_verifier_rejects_incomplete_domain_coverage():
    x = Variable()
    expr = x[0] ** 2
    problem = Problem(lambda z: z[0] ** 2, (-1.0, 1.0), 1, certified_expression=expr)
    result = minimize(problem, mode="certified", absolute_gap=0.1, max_evaluations=100)
    forged = replace(
        result.certificate,
        incumbent_x=[1.0],
        upper_bound=1.1,
        lower_bound=0.97,
        absolute_gap=0.2,
        active_regions=[{"lo": [0.99], "hi": [1.0], "lower_bound": 0.97}],
    )
    assert not verify_certificate(forged)


def test_cancellation_uses_point_interval_upper_bound():
    x = Variable()
    expr = (x[0] + 1e16) - 1e16
    problem = Problem(lambda z: z[0], (0.5, 1.5), 1, certified_expression=expr)
    result = minimize(problem, mode="certified", absolute_gap=1.0, max_evaluations=1)
    assert result.upper_bound >= 0.5
    assert float(expr.interval(result.x, result.x).hi) <= result.upper_bound


def test_arbitrary_jax_fails_closed():
    p = Problem(lambda x: jnp.sin(x[0]), (-1.0, 1.0), 1)
    with pytest.raises(ValueError, match="requires a neuralopt Expr"):
        minimize(p, mode="certified")


def test_baseline_accounting_and_certificate_coverage_tamper():
    p = Problem(
        lambda x: jnp.sum((x - 0.1) ** 2),
        (-1.0, 1.0),
        2,
        metadata={"known_optimum": 0.0},
    )
    row = run_baseline(p, "adam", 6)
    assert row["objective_evaluations"] == 6
    assert row["gradient_evaluations"] == 5

    unknown = Problem(lambda x: jnp.sum((x - 0.2) ** 2), (-1.0, 1.0), 2)
    unknown_row = run_baseline(unknown, "random", 3)
    assert unknown_row["known_optimum"] is None
    assert unknown_row["regret"] is None

    partial_domain = Problem(
        lambda x: jnp.log(x[0]),
        (-1.0, 1.0),
        1,
        x0=jnp.array([0.5]),
    )
    finite_row = run_baseline(partial_domain, "random", 8, seed=0)
    assert np.isfinite(finite_row["best_objective"])
    assert np.asarray(finite_row["final_solution"])[0] > 0
    all_nonfinite = Problem(lambda x: jnp.asarray(jnp.nan), (-1.0, 1.0), 1)
    with pytest.raises(FloatingPointError, match="no finite"):
        run_baseline(all_nonfinite, "random", 3)

    pytest.importorskip("scipy")
    limited = run_baseline(p, "scipy-lbfgsb", 1)
    assert limited["objective_evaluations"] == 1
    assert limited["status"] == "budget_exhausted"

    x = Variable()
    cp = Problem(
        lambda z: (z[0] - 0.1) ** 2,
        (-1.0, 1.0),
        1,
        certified_expression=(x[0] - 0.1) ** 2,
    )
    cert = minimize(cp, mode="certified", absolute_gap=0.1, max_evaluations=100).certificate
    assert cert is not None and verify_certificate(cert)
    cert.active_regions.pop()
    assert not verify_certificate(cert)


def test_alternative_certification_priority_tracks_true_lower_bound():
    x = Variable()
    expr = (x[0] - 0.2) ** 2 + (x[1] + 0.3) ** 2
    cert = certify(
        expr,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        gap=0.05,
        max_nodes=500,
        problem_id="alternate-priority",
        priority="widest",
        split="round_robin",
    )
    assert cert.status == "certified_gap"
    assert verify_certificate(cert)


@pytest.mark.parametrize("algorithm", ["random", "adam", "scipy-lbfgsb"])
def test_baselines_preserve_jax_objective_inputs(algorithm):
    if algorithm == "scipy-lbfgsb":
        pytest.importorskip("scipy")

    def objective(x):
        return jnp.sum(x.at[0].set(0.0) ** 2)

    def gradient(x):
        return jnp.zeros_like(x).at[1].set(2.0 * x[1])

    problem = Problem(
        objective,
        (-1.0, 1.0),
        2,
        x0=jnp.array([0.5, 0.5]),
        gradient=gradient,
        metadata={"known_optimum": 0.0},
    )
    row = run_baseline(problem, algorithm, budget=4, seed=2)
    assert 1 <= row["objective_evaluations"] <= 4
    assert np.isfinite(row["best_objective"])
