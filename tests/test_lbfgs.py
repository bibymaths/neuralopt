"""v0.8 owned L-BFGS: classical, pure JAX, projected box handling, every call counted.

Acceptance tests from docs/plans/neuralopt-v08-amortized-families.md section 3.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt.lbfgs import STATUS_NAMES, LBFGSConfig, lbfgs_solve, lbfgs_solve_batch
from neuralopt.task_families import make_task, quadratic_objective


@pytest.fixture(autouse=True)
def _float64():
    """float64 for every test here, without leaking x64 into other test modules."""
    with jax.enable_x64(True):
        yield


def _quadratic(d, kappa, seed=0):
    key = jax.random.PRNGKey(seed)
    q, _ = jnp.linalg.qr(jax.random.normal(key, (d, d), dtype=jnp.float64))
    matrix = q @ jnp.diag(jnp.geomspace(1.0, kappa, d)) @ q.T
    center = jax.random.uniform(jax.random.fold_in(key, 1), (d,), jnp.float64, -1.0, 1.0)

    def f(x):
        return 0.5 * (x - center) @ matrix @ (x - center)

    return f, matrix, center


def _projected_gradient_norm(x, g, lo, hi):
    return float(np.max(np.abs(np.clip(x - g, lo, hi) - x)))


@pytest.mark.parametrize("d", [2, 8, 32])
@pytest.mark.parametrize("kappa", [1.0, 1e2, 1e4])
def test_unconstrained_quadratic_reaches_the_analytic_minimizer(d, kappa):
    f, _, center = _quadratic(d, kappa)
    lo, hi = -5.0 * jnp.ones(d), 5.0 * jnp.ones(d)
    out = lbfgs_solve(f, jnp.zeros(d), (lo, hi), 2000)
    assert STATUS_NAMES[out["status"]] == "converged_gtol"
    assert out["projected_gradient_norm"] <= 1e-8
    np.testing.assert_allclose(out["x"], center, atol=1e-6 * kappa)
    assert out["used"] <= 2000


def test_stored_secant_pairs_satisfy_y_equals_h_s_on_a_quadratic():
    f, matrix, _ = _quadratic(8, 1e2)
    out = lbfgs_solve(f, jnp.zeros(8), (-5 * jnp.ones(8), 5 * jnp.ones(8)), 60)
    valid = np.asarray(out["pair_valid"])
    assert valid.sum() >= 3
    S, Y = np.asarray(out["S"])[valid], np.asarray(out["Y"])[valid]
    np.testing.assert_allclose(Y, S @ np.asarray(matrix).T, rtol=1e-10, atol=1e-12)


# --- SciPy reference at matched budget (harness-only accounting: each call = f + g) ---


def _scipy_lbfgsb(objective, theta, x0, lo, hi, budget):
    from scipy.optimize import minimize as scipy_minimize

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    trace = []  # (units used, best value so far), one entry per call

    class Exhausted(Exception):
        pass

    def fun(x):
        if 2 * (len(trace) + 1) > budget:
            raise Exhausted
        v, g = value_and_grad(jnp.asarray(x), theta)
        best = min(float(v), trace[-1][1] if trace else np.inf)
        trace.append((2 * (len(trace) + 1), best))
        return float(v), np.asarray(g, dtype=float)

    try:
        scipy_minimize(fun, np.asarray(x0), jac=True, method="L-BFGS-B",
                       bounds=list(zip(np.asarray(lo), np.asarray(hi), strict=True)),
                       options={"maxfun": budget // 2, "maxiter": budget, "ftol": 0.0,
                                "gtol": 0.0})
    except Exhausted:
        pass
    return trace


def _units_to(trace, target):
    for units, value in trace:
        if value <= target:
            return units
    return None


# Regret as in the v0.7 engine benchmark (results/engine-benchmark-v0.7.0.json):
# relative to the start, log10(f / f0 + 1e-16), with targets 1e-3 and 1e-6 on f / f0.
def _relative(value, f0):
    return max(value, 0.0) / f0


def _compare_with_scipy(tasks, budget):
    owned_regret, scipy_regret, owned_units, scipy_units = [], [], [], []
    for t in tasks:
        f0 = float(quadratic_objective(t.x0, t.theta))
        out = lbfgs_solve(quadratic_objective, t.x0, (t.lo, t.hi), budget, theta=t.theta)
        owned_regret.append(_relative(float(out["fun"]), f0))
        trace = [(u, _relative(v, f0)) for u, v in zip(
            out["accepted_units"][: out["accepted"]],
            np.minimum.accumulate(out["accepted_values"][: out["accepted"]]), strict=True)]
        owned_units.append(_units_to(trace, 1e-6))
        ref = _scipy_lbfgsb(quadratic_objective, t.theta, t.x0, t.lo, t.hi, budget)
        scipy_regret.append(_relative(ref[-1][1], f0))
        scipy_units.append(_units_to([(u, _relative(v, f0)) for u, v in ref], 1e-6))
    log = lambda r: np.log10(np.asarray(r) + 1e-16)  # noqa: E731
    owned_hits = sum(r <= 1e-6 for r in owned_regret)
    scipy_hits = sum(r <= 1e-6 for r in scipy_regret)
    assert owned_hits >= scipy_hits - 1, (owned_hits, scipy_hits)
    assert np.median(log(owned_regret)) <= np.median(log(scipy_regret)) + 1.0
    reached = [u for u in owned_units if u is not None]
    reached_ref = [u for u in scipy_units if u is not None]
    if reached_ref:
        assert reached and np.median(reached) <= 1.5 * np.median(reached_ref)
    return owned_hits, scipy_hits, owned_units, scipy_units, owned_regret, scipy_regret


def test_matches_scipy_on_the_v07_engine_benchmark_tasks():
    """32 rotated kappa=100 quadratics, d=16, 100 units (results/engine-benchmark-v0.7.0)."""
    tasks = [make_task("rotated", 100.0, 16, 5000 + i) for i in range(32)]
    _compare_with_scipy(tasks, 100)


@pytest.mark.parametrize(("condition", "dimension"), [(100.0, 7), (1000.0, 16), (1000.0, 32)])
def test_matches_scipy_on_rotated_curvature_tasks(condition, dimension):
    """v0.7 curvature-domain families (rotated, unseen dimensions); validation-range seeds."""
    tasks = [make_task("rotated", condition, dimension, 8500 + i) for i in range(12)]
    _compare_with_scipy(tasks, 100)


# --- bounded problems ---------------------------------------------------------------


@pytest.mark.parametrize("case", ["face", "edge", "corner"])
def test_bounded_quadratic_reaches_the_projected_minimizer(case):
    d = 4
    weights = jnp.asarray([1.0, 10.0, 100.0, 3.0])
    outside = {"face": [2.0, 0.1, -0.2, 0.3], "edge": [2.0, -3.0, 0.2, 0.0],
               "corner": [2.0, -3.0, 4.0, -2.5]}[case]
    center = jnp.asarray(outside)

    def f(x):
        return 0.5 * jnp.sum(weights * (x - center) ** 2)

    lo, hi = -jnp.ones(d), jnp.ones(d)
    out = lbfgs_solve(f, jnp.zeros(d), (lo, hi), 400)
    expected = np.clip(np.asarray(center), -1.0, 1.0)  # separable: exact projected solution
    np.testing.assert_allclose(out["x"], expected, atol=1e-8)
    assert out["projected_gradient_norm"] <= 1e-8
    active = np.isclose(out["x"], -1.0) | np.isclose(out["x"], 1.0)
    assert (active == (np.abs(np.asarray(center)) > 1.0)).all()


def test_bounded_rotated_quadratic_satisfies_kkt_and_stays_in_the_box():
    f, _, _ = _quadratic(6, 1e2, seed=3)
    lo, hi = -0.2 * jnp.ones(6), 0.1 * jnp.ones(6)
    out = lbfgs_solve(f, jnp.zeros(6), (lo, hi), 1000)
    assert out["projected_gradient_norm"] <= 1e-8
    assert np.all(out["x"] >= -0.2) and np.all(out["x"] <= 0.1)
    assert np.all(out["trial_min"] >= -0.2) and np.all(out["trial_max"] <= 0.1)


def test_clipped_rosenbrock_reaches_a_kkt_point():
    def f(x):
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    lo, hi = jnp.asarray([-2.0, -2.0]), jnp.asarray([0.5, 2.0])  # unconstrained optimum (1, 1)
    out = lbfgs_solve(f, jnp.asarray([-1.5, 1.5]), (lo, hi), 2000)
    assert out["projected_gradient_norm"] <= 1e-8
    np.testing.assert_allclose(out["x"], [0.5, 0.25], atol=1e-6)


# --- accounting ---------------------------------------------------------------------


def _counted(f):
    calls = {"n": 0}

    def wrapped(x):
        jax.debug.callback(lambda: calls.__setitem__("n", calls["n"] + 1), ordered=True)
        return f(x)

    return wrapped, calls


def test_counted_calls_equal_callback_executions_and_stay_within_budget():
    f, _, _ = _quadratic(5, 1e3, seed=2)
    wrapped, calls = _counted(f)
    budget = 37
    out = lbfgs_solve(wrapped, jnp.zeros(5), (-5 * jnp.ones(5), 5 * jnp.ones(5)), budget,
                      config=LBFGSConfig(gtol=0.0))
    # every objective call and every gradient call runs the objective's forward pass once
    assert calls["n"] == out["objective_evaluations"] + out["gradient_evaluations"]
    assert out["used"] == out["objective_evaluations"] + out["gradient_evaluations"]
    assert budget - 1 <= out["used"] <= budget
    assert out["gradient_evaluations"] == 1 + out["accepted"]


def test_nonfinite_trials_are_rejected_and_counted():
    def f(x):  # nan beyond x0 > 0.5: the first full step lands there
        return jnp.where(x[0] > 0.5, jnp.nan, (x[0] - 2.0) ** 2 + x[1] ** 2)

    out = lbfgs_solve(f, jnp.asarray([0.0, 0.0]), (-3 * jnp.ones(2), 3 * jnp.ones(2)), 60)
    assert out["nonfinite_trials"] >= 1
    assert np.isfinite(out["fun"])
    assert 0.25 <= float(out["x"][0]) <= 0.5  # approached the region edge from the finite side


def test_nonfinite_start_is_reported_without_iterating():
    out = lbfgs_solve(lambda x: jnp.sum(x) * jnp.nan, jnp.zeros(2),
                      (-jnp.ones(2), jnp.ones(2)), 20)
    assert STATUS_NAMES[out["status"]] == "nonfinite_start"
    assert out["used"] == 2 and out["accepted"] == 0


# --- parity and compilation -----------------------------------------------------------


def test_batched_solves_equal_single_solves():
    tasks = [make_task("rotated", 100.0, 8, 7000 + i) for i in range(5)]
    thetas = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[t.theta for t in tasks])
    x0 = jnp.stack([t.x0 + 0.1 * i for i, t in enumerate(tasks)])
    lo, hi = jnp.stack([t.lo for t in tasks]), jnp.stack([t.hi for t in tasks])
    batch = lbfgs_solve_batch(quadratic_objective, thetas, x0, (lo, hi), 80)
    for i, t in enumerate(tasks):
        single = lbfgs_solve(quadratic_objective, x0[i], (t.lo, t.hi), 80, theta=t.theta)
        for key in ("objective_evaluations", "gradient_evaluations", "accepted", "status"):
            assert int(batch[key][i]) == int(single[key]), key
        np.testing.assert_allclose(batch["x"][i], single["x"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(batch["accepted_values"][i], single["accepted_values"],
                                   rtol=1e-12, atol=1e-14)


def test_new_context_of_the_same_shape_does_not_recompile():
    a, b = make_task("rotated", 100.0, 8, 1), make_task("rotated", 100.0, 8, 2)
    lbfgs_solve(quadratic_objective, a.x0, (a.lo, a.hi), 50, theta=a.theta)
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture()
    logging.getLogger("jax").addHandler(handler)
    try:
        with jax.log_compiles(True):
            lbfgs_solve(quadratic_objective, b.x0, (b.lo, b.hi), 50, theta=b.theta)
    finally:
        logging.getLogger("jax").removeHandler(handler)
    assert not [m for m in messages if "Compiling" in m]
