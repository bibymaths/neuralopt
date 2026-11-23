"""Statistics and decision rules of the v0.8 main-track evaluation (pre-registration §7, §8,
amendments 1, 2 and 4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_stats_v0_8 as st  # noqa: E402


def test_time_area_counts_grid_points_at_or_after_the_cost():
    # T = 80 s: grid 1, 2, ..., 80 s
    assert st.time_area(0.5, 80.0) == 1.0
    assert st.time_area(40.0, 80.0) == pytest.approx(41 / 80)
    assert st.time_area(80.0, 80.0) == pytest.approx(1 / 80)
    assert st.time_area(80.5, 80.0) == 0.0
    assert st.time_area(np.inf, 80.0) == 0.0


def test_unit_area_matches_amendment_1():
    assert st.unit_area(None) == 0.0
    assert st.unit_area(10) == 1.0
    assert st.unit_area(15) == pytest.approx(79 / 80)
    assert st.unit_area(800) == pytest.approx(1 / 80)
    assert st.unit_area(801) == 0.0


def test_success_within_and_sequential_multistart():
    assert st.success_within([10, None, 300], 200).tolist() == [True, False, False]
    # starts run one after another with a per-start cap of 100 units
    assert st.sequential_cost([None, 40, 5], 100) == 140
    assert st.sequential_cost([150, 40], 100) == 140
    assert st.sequential_cost([None] * 16, 100) is None


def test_crossed_bootstrap_resamples_seeds_and_instances():
    rng = np.random.default_rng(1)
    m = rng.normal(0.3, 1.0, (5, 400))
    out = st.crossed_bootstrap(m, draws=2000, seed=0)
    assert out["mean"] == pytest.approx(m.mean())
    lo, hi = out["ci95"]
    assert lo < m.mean() < hi
    assert 0.1 < hi - lo < 0.5
    # a single row is an instance-only bootstrap, reproducible from the seed
    one = st.crossed_bootstrap(m[:1], draws=2000, seed=0)
    assert one == st.crossed_bootstrap(m[:1], draws=2000, seed=0)


def test_crossed_bootstrap_widens_with_seed_spread():
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 1.0, 300)
    tight = np.tile(base, (5, 1))
    spread = tight + np.array([-1.0, -0.5, 0.0, 0.5, 1.0])[:, None]
    w_tight = np.diff(st.crossed_bootstrap(tight, draws=2000)["ci95"])[0]
    w_spread = np.diff(st.crossed_bootstrap(spread, draws=2000)["ci95"])[0]
    assert w_spread > 2 * w_tight


def test_clopper_pearson_known_values_and_fractional_counts():
    lo, hi = st.clopper_pearson(0, 300)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 300), rel=1e-9)
    lo, hi = st.clopper_pearson(300, 300)
    assert hi == 1.0
    a = st.clopper_pearson(3.0, 1000)
    b = st.clopper_pearson(3.4, 1000)
    assert a[1] < b[1]


def test_auroc_with_ties():
    assert st.auroc(np.array([0.1, 0.2]), np.array([0.3, 0.4])) == 1.0
    assert st.auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5
    assert st.auroc(np.array([0.1, 0.4]), np.array([0.2, 0.3])) == 0.5


def test_decisions_c1_c6_c7_shape():
    assert st.decide_superiority([0.01, 0.05]) == "supported"
    assert st.decide_superiority([-0.05, -0.0]) == "not supported"
    assert st.decide_superiority([-0.01, 0.02]) == "inconclusive"


def test_decision_c3_better_on_one_axis_and_non_inferior_on_the_other():
    assert st.decide_c3([0.01, 0.03], [-0.015, 0.01]) == "supported"
    assert st.decide_c3([-0.015, 0.01], [0.001, 0.02]) == "supported"
    # neither axis better
    assert st.decide_c3([-0.01, 0.01], [-0.01, 0.02]) == "not supported"
    # better on one axis, inferior on the other (interval entirely below -0.02)
    assert st.decide_c3([0.01, 0.03], [-0.06, -0.03]) == "not supported"
    # better on one axis, the other neither non-inferior nor inferior
    assert st.decide_c3([0.01, 0.03], [-0.04, 0.0]) == "inconclusive"


def test_decision_c5():
    assert st.decide_c5([-0.005, 0.01], [0.0, 0.008]) == "supported"
    assert st.decide_c5([-0.03, -0.015], [0.0, 0.008]) == "not supported"
    assert st.decide_c5([-0.005, 0.01], [0.012, 0.03]) == "not supported"
    assert st.decide_c5([-0.02, 0.01], [0.0, 0.008]) == "inconclusive"


def test_decision_all_bars_intersection_union():
    # C2 (amended): every bar on both axes needs lower bound > 0
    good = [0.01, 0.05]
    assert st.decide_all_better([good, good, good, good]) == "supported"
    assert st.decide_all_better([good, [-0.05, -0.01], good]) == "not supported"
    assert st.decide_all_better([good, [-0.01, 0.02]]) == "inconclusive"


def test_decision_c4_bars():
    assert st.decide_c4([[0.1, 0.5], [0.2, 0.3]]) == "supported"
    assert st.decide_c4([[0.1, 0.5], [-0.3, -0.1]]) == "not supported"
    assert st.decide_c4([[0.1, 0.5], [-0.3, 0.1]]) == "inconclusive"


def test_decision_c8_c9_c10():
    assert st.decide_c8([[0.55, 0.7], [0.6, 0.8]]) == "supported"
    assert st.decide_c8([[0.55, 0.7], [0.3, 0.45]]) == "not supported"
    assert st.decide_c8([[0.45, 0.7]]) == "inconclusive"
    assert st.decide_c9([0.85, 1.1]) == "supported"
    assert st.decide_c9([0.5, 0.75]) == "not supported"
    assert st.decide_c9([0.7, 0.9]) == "inconclusive"
    assert st.decide_c10([0.01, 0.03]) == "supported"
    assert st.decide_c10([0.2, 0.4]) == "not supported"
    assert st.decide_c10([0.03, 0.07]) == "inconclusive"


def test_b95_on_the_f0_grid():
    units = [20] * 95 + [None] * 5
    assert st.b95(units) == 25
    assert st.b95([None] * 10) is None


def test_logistic_fit_recovers_a_slope():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 2, 4000)
    p = 1 / (1 + np.exp(-(2.0 - 3.0 * x)))
    y = rng.uniform(size=x.size) < p
    a, b = st.logistic_fit(x, y)
    assert a == pytest.approx(2.0, abs=0.3)
    assert b == pytest.approx(-3.0, abs=0.3)
