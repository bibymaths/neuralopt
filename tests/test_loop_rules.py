"""v0.7 step-acceptance rules (opt-in ``loop_rule``) for the compiled preconditioned loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import Problem, minimize
from neuralopt.compiled import preconditioned_solve_batch
from neuralopt.features import gradient_to_normalized, normalize_coordinates
from neuralopt.gram import GramConfig, init_gram_model
from neuralopt.loop_rules import C1, C2, LOOP_RULES
from neuralopt.model import ModelConfig, init_model

HISTORY = ModelConfig(width=16, heads=4, rank=2, layers=1, scale_aware=True,
                      geometry_bound=10.0, lowrank_scale=0.5, feature_set="history-v1",
                      history_length=4)
GRAM = GramConfig(history_length=4, width=16, heads=4, layers=1)


def _rotated(d=6, seed=0, budget=80, bound=3.0):
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (d, d)))
    hessian = q @ jnp.diag(jnp.geomspace(1.0, 100.0, d)) @ q.T
    center = 0.3 * jax.random.normal(jax.random.PRNGKey(seed + 1), (d,))

    def objective(x):
        return 0.5 * (x - center) @ hessian @ (x - center)

    return Problem(objective, (-bound, bound), d, x0=np.full(d, 1.5), max_evaluations=budget,
                   dtype=jnp.float64)


def _solve(problem, rule, geometry="fixed", params=None, config=HISTORY, model_kind="transformer",
           step=0.05, engine="compiled"):
    return minimize(problem, params=params, config=config, model_kind=model_kind,
                    candidate_method="preconditioned", hamiltonian_metric=geometry,
                    fixed_metric_diag=np.ones(problem.dimension), preconditioned_step=step,
                    record_diagnostics=True, preconditioned_engine=engine, loop_rule=rule)


def test_rule_names_and_default():
    assert LOOP_RULES == ("halving", "armijo", "wolfe")
    with jax.enable_x64(True):
        problem = _rotated()
        default = minimize(problem, candidate_method="preconditioned", hamiltonian_metric="fixed",
                           fixed_metric_diag=np.ones(6), preconditioned_step=0.05,
                           preconditioned_engine="compiled")
        explicit = _solve(problem, "halving")
    np.testing.assert_array_equal(default.trajectory, explicit.trajectory)
    assert explicit.diagnostics["loop_rule"] == "halving"


@pytest.mark.parametrize("rule", ["armijo", "wolfe"])
def test_python_engine_only_supports_the_v06_rule(rule):
    with jax.enable_x64(True), pytest.raises(ValueError, match="loop_rule"):
        _solve(_rotated(), rule, engine="python")


@pytest.mark.parametrize("rule", LOOP_RULES)
def test_budget_and_unit_accounting_are_exact(rule):
    with jax.enable_x64(True):
        result = _solve(_rotated(budget=73), rule)
    d = result.diagnostics
    trials = sum(r["trials"] for r in d["records"])
    accepted = sum(r["accepted"] for r in d["records"])
    assert d["objective_evaluations"] + d["gradient_evaluations"] == result.evaluations <= 73
    assert d["objective_evaluations"] == 1 + trials
    if rule == "wolfe":  # every trial evaluates the objective and its gradient
        assert d["gradient_evaluations"] == 1 + trials
    else:  # a gradient only at an improving trial
        assert d["gradient_evaluations"] >= 1 + accepted
        assert d["gradient_evaluations"] <= 1 + accepted + (0 if rule == "armijo" else trials)
    assert d["accepted_steps"] == accepted


def _incumbents(problem, result):
    """(x, f, g_u) before each iteration, from the recorded accepted points."""
    lo, hi = problem.bounds
    x, out = problem.x0, []
    for record in result.diagnostics["records"]:
        g = jax.grad(problem.objective)(x)
        out.append((x, float(problem.objective(x)), gradient_to_normalized(g, lo, hi)))
        if record["accepted"]:
            x = jnp.asarray(record["candidate"])
    return out


def test_armijo_accepts_only_sufficient_decrease():
    with jax.enable_x64(True):
        problem = _rotated(seed=2)
        result = _solve(problem, "armijo", step=0.5)
        lo, hi = problem.bounds
        checked = 0
        for (x, f, g_u), r in zip(_incumbents(problem, result), result.diagnostics["records"],
                                  strict=True):
            if r["accepted"]:
                du = normalize_coordinates(jnp.asarray(r["candidate"]), lo, hi) - \
                    normalize_coordinates(x, lo, hi)
                assert r["candidate_objective"] < f
                assert r["candidate_objective"] <= f + C1 * float(g_u @ du) + 1e-12
                checked += 1
        assert checked >= 5
        assert any(r["trials"] > 1 for r in result.diagnostics["records"])  # backtracking ran


def test_wolfe_lengthens_short_steps_and_satisfies_strong_wolfe():
    with jax.enable_x64(True):
        problem = _rotated(seed=3)
        result = _solve(problem, "wolfe", step=0.02)  # too short, within the 10x lengthening cap
        lo, hi = problem.bounds
        records = result.diagnostics["records"]
        assert any(r["step_length"] > 0.02 * 1.5 for r in records)  # lengthened
        satisfied = 0
        for (x, f, g_u), r in zip(_incumbents(problem, result), records, strict=True):
            if not r["accepted"] or not r["wolfe_satisfied"]:
                continue
            xn = jnp.asarray(r["candidate"])
            direction = normalize_coordinates(xn, lo, hi) - normalize_coordinates(x, lo, hi)
            g_new = gradient_to_normalized(jax.grad(problem.objective)(xn), lo, hi)
            slope0, slope1 = float(g_u @ direction), float(g_new @ direction)
            assert r["candidate_objective"] <= f + C1 * slope0 + 1e-12
            assert abs(slope1) <= C2 * abs(slope0) + 1e-12
            satisfied += 1
        assert satisfied >= 3


@pytest.mark.parametrize("rule", LOOP_RULES)
def test_exact_model_step_is_accepted_unchanged(rule):
    d = 3
    with jax.enable_x64(True):
        problem = Problem(lambda x: jnp.sum(x**2), (-1.0, 1.0), d, x0=np.full(d, 0.5),
                          max_evaluations=10, dtype=jnp.float64)
        # Normalized gradient times L = ‖x0‖ lands exactly on the minimizer.
        result = _solve(problem, rule, step=float(np.sqrt(d) * 0.5))
    first = result.diagnostics["records"][0]
    assert first["accepted"] and first["initial_accepted"] and first["trials"] == 1
    assert first["candidate_objective"] == pytest.approx(0.0, abs=1e-24)


@pytest.mark.parametrize("rule", ["armijo", "wolfe"])
@pytest.mark.parametrize("model", ["history", "gram"])
def test_new_rules_keep_the_history_coherent(rule, model):
    with jax.enable_x64(True):
        problem = _rotated(seed=4, budget=60)
        if model == "history":
            params, config, kind = init_model(jax.random.PRNGKey(0), HISTORY), HISTORY, \
                "transformer"
        else:
            params, config, kind = init_gram_model(jax.random.PRNGKey(0), GRAM), GRAM, "gram"
        result = _solve(problem, rule, geometry="learned", params=params, config=config,
                        model_kind=kind)
        hist = result.diagnostics["history"]
        lo, hi = problem.bounds
        valid = np.asarray(hist["valid"])
        assert valid.sum() >= 2
        for u, g, f in zip(np.asarray(hist["u"])[valid], np.asarray(hist["g"])[valid],
                           np.asarray(hist["f"])[valid], strict=True):
            x = lo + 0.5 * (jnp.asarray(u) + 1.0) * (hi - lo)
            np.testing.assert_allclose(f, problem.objective(x), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(g, gradient_to_normalized(jax.grad(problem.objective)(x),
                                                                 lo, hi), rtol=1e-8, atol=1e-10)


def _parametric(x, theta):
    center, matrix = theta
    return 0.5 * (x - center) @ matrix @ (x - center)


@pytest.mark.parametrize("rule", ["armijo", "wolfe"])
def test_batched_rules_match_single_solves(rule):
    d, tasks, budget = 4, 4, 50
    with jax.enable_x64(True):
        keys = jax.random.split(jax.random.PRNGKey(9), tasks)
        qs = jax.vmap(lambda k: jnp.linalg.qr(jax.random.normal(k, (d, d)))[0])(keys)
        matrices = jnp.einsum("bij,j,bkj->bik", qs, jnp.geomspace(1.0, 50.0, d), qs)
        centers = jnp.zeros((tasks, d))
        lo, hi = -2.0 * jnp.ones((tasks, d)), 2.0 * jnp.ones((tasks, d))
        x0 = jnp.full((tasks, d), 1.0)
        batch = preconditioned_solve_batch(_parametric, (centers, matrices), x0, (lo, hi),
                                           budget, geometry="fixed",
                                           fixed_metric_diag=np.ones(d),
                                           preconditioned_step=0.1, loop_rule=rule)
        for i in range(tasks):
            single = minimize(
                Problem(lambda x, i=i: _parametric(x, (centers[i], matrices[i])), (lo[i], hi[i]),
                        d, x0=np.asarray(x0[i]), dtype=jnp.float64),
                max_evaluations=budget, candidate_method="preconditioned",
                hamiltonian_metric="fixed", fixed_metric_diag=np.ones(d),
                preconditioned_step=0.1, preconditioned_engine="compiled", loop_rule=rule,
            )
            assert int(batch["objective_evaluations"][i]) == \
                single.diagnostics["objective_evaluations"]
            assert int(batch["gradient_evaluations"][i]) == \
                single.diagnostics["gradient_evaluations"]
            np.testing.assert_allclose(float(batch["fun"][i]), single.fun, rtol=1e-10)
