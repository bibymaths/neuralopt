"""v0.8-B harness baselines and metrics: call accounting, traces, anytime regret, polish."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt.families import SYNTHETIC, family_objective, make_split

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("cma")
pytest.importorskip("optuna")


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


def _module():
    spec = importlib.util.spec_from_file_location("v08b_baselines",
                                                  ROOT / "scripts" / "v08b_baselines.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["v08b_baselines"] = module
    spec.loader.exec_module(module)
    return module


NAME = "rosenbrock-shifted"


def _instance(n=1):
    contexts, truths = make_split(NAME, "validation", n)
    lo, hi = (np.asarray(b) for b in SYNTHETIC[NAME].bounds("validation"))
    return contexts, truths, lo, hi


class _Counter:
    """Counts objective calls of a single-point or batched objective on one instance."""

    def __init__(self, contexts, batched):
        self.c = jax.tree_util.tree_map(lambda a: a[0], contexts)
        self.f = jax.jit(family_objective(NAME))
        self.fb = jax.jit(jax.vmap(family_objective(NAME), in_axes=(0, None)))
        self.vg = jax.jit(jax.value_and_grad(family_objective(NAME)))
        self.calls = 0
        self.points: list[np.ndarray] = []
        self.batched = batched

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        self.calls += x.shape[0]
        self.points.extend(x)
        return np.asarray(self.fb(jnp.asarray(x), self.c))

    def single(self, x):
        return float(self(x)[0])

    def value_and_grad(self, x):
        self.calls += 1
        self.points.append(np.asarray(x, float))
        v, g = self.vg(jnp.asarray(x), self.c)
        return float(v), np.asarray(g)


def test_best_so_far_and_anytime_regret_follow_the_plan_definition():
    m = _module()
    trace = m.make_trace([2, 5], [0.1, 1e-8], np.zeros((2, 1)), objective_calls=2,
                         gradient_calls=0)
    fbest = m.best_so_far(trace, 6)
    assert np.isinf(fbest[0]) and fbest[1] == 0.1 and fbest[4] == 1e-8
    curve = m.clipped_log_regret(fbest, f_star=0.0, tau=1e-6, scale=1.0)
    np.testing.assert_allclose(curve, [1, -1, -1, -1, -6, -6])
    assert m.anytime_auc(trace, 6, 0.0, 1e-6, 1.0) == pytest.approx(-14 / 6)
    capped = m.make_trace([1], [100.0], np.zeros((1, 1)), 1, 0)
    assert m.anytime_auc(capped, 1, 0.0, 1e-6, 1.0) == pytest.approx(1.0)  # cap 10·s
    empty = m.make_trace([], [], np.zeros((0, 1)), 0, 0)
    assert m.anytime_auc(empty, 4, 0.0, 1e-6, 1.0) == pytest.approx(1.0)
    assert m.best_point(trace, 4) is not None and m.best_point(empty, 4) is None


def test_multistart_concatenates_starts_at_per_start_offsets():
    m = _module()
    a = m.make_trace([1, 3, 9], [3.0, 2.0, 0.5], np.arange(3)[:, None] * 1.0, 3, 0)
    b = m.make_trace([1, 2], [1.0, 0.7], np.arange(2)[:, None] * 1.0, 2, 0)
    combined = m.combine_multistart([a, b], per_start=4)
    np.testing.assert_array_equal(combined["units"], [1, 3, 5, 6])
    np.testing.assert_allclose(combined["values"], [3.0, 2.0, 1.0, 0.7])


@pytest.mark.parametrize("arm", ["quasi_random", "cmaes", "tpe"])
def test_gradient_free_arms_charge_one_unit_per_objective_call_and_start_at_x0(arm):
    m = _module()
    contexts, _, lo, hi = _instance()
    x0 = np.asarray(contexts["x0"][0])
    counter = _Counter(contexts, batched=True)
    budget = 30
    if arm == "quasi_random":
        trace = m.quasi_random(counter, x0, lo, hi, budget, seed=3)
    elif arm == "cmaes":
        trace = m.cmaes(counter, x0, lo, hi, budget, sigma0=0.2, popsize=None, seed=3)
    else:
        trace = m.tpe(counter.single, x0, lo, hi, budget, n_startup=5, multivariate=False,
                      seed=3)
    assert counter.calls == trace["objective_calls"] == budget
    assert trace["gradient_calls"] == 0
    np.testing.assert_array_equal(trace["units"], np.arange(1, budget + 1))
    np.testing.assert_allclose(trace["points"][0], x0)
    assert np.all((trace["points"] >= lo) & (trace["points"] <= hi))
    np.testing.assert_allclose(trace["values"][0], counter.single(x0))
    if arm == "quasi_random":
        again = m.quasi_random(counter, x0, lo, hi, budget, seed=3)
    elif arm == "cmaes":
        again = m.cmaes(counter, x0, lo, hi, budget, sigma0=0.2, popsize=None, seed=3)
    else:
        again = m.tpe(counter.single, x0, lo, hi, budget, n_startup=5, multivariate=False,
                      seed=3)
    np.testing.assert_array_equal(again["values"], trace["values"])


def test_scipy_arm_charges_two_units_per_call_and_uses_the_full_budget():
    m = _module()
    contexts, _, lo, hi = _instance()
    x0 = np.asarray(contexts["x0"][0])
    counter = _Counter(contexts, batched=False)
    trace = m.scipy_lbfgsb(counter.value_and_grad, x0, lo, hi, 40)
    assert counter.calls == trace["objective_calls"] == trace["gradient_calls"] == 20
    np.testing.assert_array_equal(trace["units"], np.arange(2, 41, 2))
    np.testing.assert_allclose(trace["points"][0], x0)


def test_owned_lbfgs_trace_matches_the_solver_accounting():
    m = _module()
    contexts, _, lo, hi = _instance(3)
    x0 = np.asarray(contexts["x0"])
    traces, out = m.owned_lbfgs(family_objective(NAME), contexts, x0, lo, hi, 60,
                                return_raw=True)
    for i, tr in enumerate(traces):
        assert tr["units"][0] == 1  # the start's value is available after its objective call
        k = int(out["accepted"][i])
        np.testing.assert_array_equal(tr["units"][1:], np.asarray(out["accepted_units"][i][:k]) - 1)
        assert tr["objective_calls"] + tr["gradient_calls"] == int(out["used"][i]) <= 60
        assert m.best_so_far(tr, 60)[-1] == pytest.approx(float(out["fun"][i]))
        np.testing.assert_allclose(m.best_point(tr, 60), out["x"][i])


def test_bb_trace_matches_the_compiled_loop_accounting():
    m = _module()
    contexts, _, lo, hi = _instance(3)
    x0 = np.asarray(contexts["x0"])
    traces, out = m.bb(family_objective(NAME), contexts, x0, lo, hi, 50,
                       fallback_length=0.1, return_raw=True)
    for i, tr in enumerate(traces):
        assert tr["objective_calls"] + tr["gradient_calls"] == int(out["used"][i]) <= 50
        assert tr["units"][0] == 1 and np.all(np.diff(tr["units"]) > 0)
        assert tr["units"][-1] <= int(out["used"][i])
        np.testing.assert_allclose(tr["points"][0], x0[i])
        assert m.best_so_far(tr, 50)[-1] == pytest.approx(float(out["value"][i]))


def test_polish_finds_the_global_optimum_from_inside_its_basin_only():
    m = _module()
    name = "rastrigin"
    contexts, truths = make_split(name, "validation", 2)
    lo, hi = (np.asarray(b) for b in SYNTHETIC[name].bounds("validation"))
    near = np.asarray(truths) + 0.05
    far = np.clip(np.asarray(truths) + 2.0, lo, hi)
    values = m.polish(family_objective(name), contexts, np.concatenate([near, far]), lo, hi,
                      tile=2)
    assert np.all(values[:2] <= 1e-6) and np.all(values[2:] > 0.5)


def test_package_never_imports_harness_only_baseline_packages():
    for path in (ROOT / "src" / "neuralopt").rglob("*.py"):
        text = path.read_text()
        for package in ("cma", "optuna"):
            assert f"import {package}" not in text and f"from {package}" not in text, path


def test_best_point_takes_the_last_entry_among_tied_best_values():
    m = _module()
    points = np.array([[0.0], [np.nan], [2.0]])
    trace = m.make_trace([1, 2, 3], [5.0, 1.0, 1.0], points, 3, 0)
    np.testing.assert_allclose(m.best_point(trace, 3), [2.0])
