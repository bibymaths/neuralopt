"""Derived per-instance outcomes of the v0.8 evaluation analysis (amendments 1 and 4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# work in progress: the module is written next, test-first (HANDOFF.md)
an = pytest.importorskip("analyze_eval_v0_8")

INF = np.inf


def test_units_array_maps_none_to_infinity():
    np.testing.assert_array_equal(an.units([3, None]), [3.0, INF])
    np.testing.assert_array_equal(an.units([[None, 4]]), [[INF, 4.0]])


def test_sequential_costs_vectorized_match_amendment_1():
    per_start = an.units([[None, 40, 5], [150, 40, None], [None, None, None]])
    np.testing.assert_array_equal(an.sequential_costs(per_start, 100), [140, 140, INF])


def test_unit_and_time_areas_vectorized():
    np.testing.assert_allclose(an.unit_areas(np.array([10.0, 15.0, 801.0, INF])),
                               [1.0, 79 / 80, 0.0, 0.0])
    np.testing.assert_allclose(an.time_areas(np.array([0.5, 40.0, INF]), 80.0),
                               [1.0, 41 / 80, 0.0])


def test_top_k_success_for_learned_ranks_and_sobol_starts():
    # ranks: (seed, instance, rank); budget 200 -> K=4 gives 50 units per start
    rank_u = an.units([[[None, 60, 45, None, None, None, None, None],
                        [30, None, None, None, None, None, None, None]]])
    np.testing.assert_array_equal(an.top_k_success(rank_u, 4, 200), [[True, True]])
    np.testing.assert_array_equal(an.top_k_success(rank_u, 2, 200), [[True, True]])
    np.testing.assert_array_equal(an.top_k_success(rank_u, 8, 200), [[False, False]])
    sobol = an.units([[None, 49, None, None, 1], [None, None, None, None, 1]])
    np.testing.assert_array_equal(an.top_k_success(sobol, 4, 200), [True, False])


def test_a4f_success_and_silent_miss():
    b = 200
    a4 = np.array([[150.0, INF, INF, 300.0]])
    trig = np.array([[False, True, False, True]])
    a2o = an.units([[None, None], [None, 90], [None, None], [None, None]])
    # b = 400: fallback gets (400-200)//2 = 100 units per start
    np.testing.assert_array_equal(an.a4f_success(a4, trig, a2o, b, 400),
                                  [[True, True, False, False]])
    np.testing.assert_array_equal(an.silent_miss(a4, trig, b, 400),
                                  [[False, False, True, False]])
    # untriggered A4 continues past B
    np.testing.assert_array_equal(an.a4f_success(a4, trig, a2o, b, 800)[0, 3], False)
    np.testing.assert_array_equal(an.a4f_success(np.array([[300.0]]), np.array([[False]]),
                                                 an.units([[None, None]]), b, 400), [[True]])
    # below B the trigger has not been evaluated: A4 alone, and misses are silent
    np.testing.assert_array_equal(an.a4f_success(a4, trig, a2o, b, 100),
                                  [[False, False, False, False]])
    np.testing.assert_array_equal(an.silent_miss(a4, trig, b, 100), [[True, True, True, True]])


def test_b95_from_a_success_curve_and_the_c5_budget():
    curve = {25: 0.1, 50: 0.5, 100: 0.96, 200: 1.0, 400: 1.0, 800: 1.0}
    assert an.b95_of(curve) == 100
    assert an.b95_of({k: 0.5 for k in curve}) is None
    assert an.c5_budget(100, None) == 800
    assert an.c5_budget(100, 400) == 400


def test_better_multistart_on_the_wall_axis_uses_a_common_horizon():
    # A2 is slow per unit but succeeds early in units; A2o is fast per unit
    cost_a2, cost_a2o = np.array([100.0, 100.0]), np.array([400.0, 400.0])
    kappa_a2, kappa_a2o = 1e-3, 1e-4
    assert an.better_on_wall(cost_a2, kappa_a2, cost_a2o, kappa_a2o) == "A2o"
    assert an.better_on_wall(cost_a2, 1e-4, cost_a2o, 1e-4) == "A2"


def test_c9_ratio_is_top_quartile_success_over_overall():
    err = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    ok = np.array([[1, 1, 1, 1, 1, 1, 0, 1]], bool)
    # top quartile (errors 0.7, 0.8): success 0.5; overall 7/8
    assert an.c9_ratio(err, ok) == pytest.approx(0.5 / (7 / 8))


def test_c9_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(0)
    err = rng.uniform(0, 1, (5, 400))
    ok = rng.uniform(size=(5, 400)) < 0.7
    out = an.c9_interval(err, ok, draws=500)
    assert out["ci95"][0] <= out["ratio"] <= out["ci95"][1]


def test_auroc_interval_with_seed_resampling():
    rng = np.random.default_rng(1)
    neg = rng.uniform(0.5, 1.0, (5, 300))   # in-distribution confidence (high)
    pos = rng.uniform(0.0, 0.6, (5, 100))   # OOD confidence (low)
    out = an.auroc_interval(neg, pos, draws=300)
    assert out["auroc"] > 0.9 and out["ci95"][0] > 0.85
