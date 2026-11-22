"""The frozen v0.7 protocol harness: splits, decision rules, claim table, smoke run.

No test task is built here: the smoke run uses training-seed tasks and untrained models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gram_space_v0_7 as protocol  # noqa: E402

from neuralopt.gram import init_gram_model  # noqa: E402


def test_preregistration_hash_is_checked():
    assert len(protocol.check_preregistration()) == 64


def test_splits_are_disjoint_and_test_dimensions_unseen():
    train, test = protocol.training_specs(), protocol.test_specs()
    assert len(train) == 96 and len(test) == 84
    assert {s["seed"] for s in train}.isdisjoint({s["seed"] for s in test})
    assert all(7000 <= s["seed"] < 8000 for s in train)
    assert all(9000 <= s["seed"] < 10000 for s in test)
    assert {s["dimension"] for s in test}.isdisjoint({s["dimension"] for s in train})


def test_equal_spectrum_pairs_share_seed_and_spectrum():
    """Test specs are compared without building test tasks; the generator property
    (same seed -> same permuted spectrum for diagonal and rotated) is checked on seeds
    outside every split."""
    test = protocol.test_specs()
    diagonal = [s for s in test if (s["family"], s["condition"]) == ("diagonal", 1000.0)]
    rotated = [s for s in test if (s["family"], s["condition"]) == ("rotated", 1000.0)]
    assert len(diagonal) == len(rotated) == 12
    for d, r in zip(diagonal, rotated, strict=True):
        assert (d["seed"], d["dimension"], d["objective_scale"], d["coordinate_scale"]) == (
            r["seed"], r["dimension"], r["objective_scale"], r["coordinate_scale"])
    with jax.enable_x64(True):
        for seed in (123, 456):
            a = protocol.make_task("diagonal", 1000.0, 7, seed)
            b = protocol.make_task("rotated", 1000.0, 7, seed)
            np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(np.asarray(a.theta["matrix"]))),
                                       np.sort(np.linalg.eigvalsh(np.asarray(b.theta["matrix"]))),
                                       rtol=1e-10)


def test_training_groups_are_cycled_in_proportion_to_their_size():
    groups = protocol.training_groups()
    names = [g.name for g in groups]
    assert len(set(names)) == 6
    for dim in (4, 8, 12):
        assert names.count(f"quadratic-d{dim}") == 7 * names.count(f"rastrigin-d{dim}")


def test_decision_rules():
    def interval(lo, hi, mean=None):
        return {"mean": (lo + hi) / 2 if mean is None else mean, "ci95": [lo, hi]}

    assert protocol.superiority(interval(-0.5, -0.01))
    assert not protocol.superiority(interval(-0.5, 0.01))
    assert protocol.non_inferior(interval(-0.3, 0.24), 0.25)
    assert not protocol.non_inferior(interval(-0.3, 0.26), 0.25)
    assert protocol.equivalent(interval(-0.49, 0.49))
    assert not protocol.equivalent(interval(-0.51, 0.2))
    assert protocol.guard_violated({"ackley": interval(0.1, 1.2, mean=0.6)})
    assert not protocol.guard_violated({"ackley": interval(-0.1, 1.2, mean=0.6)})
    assert not protocol.guard_violated({"ackley": interval(0.1, 0.8, mean=0.45)})


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ((False,), "round ends"),
        ((True, False, False, False, False, False), "did not beat BB"),
        ((True, True, True, True, True, True), "No superiority claim"),
        ((True, True, False, False, None, True), "short of the oracle-BFGS ceiling"),
        ((True, True, False, False, None, False), "no rotation-invariance claim"),
        ((True, True, False, True, True, True), "relearning L-BFGS"),
        ((True, True, False, True, True, False), "relearning L-BFGS"),
        ((True, True, False, True, False, True), "not established"),
        ((True, True, False, True, False, False), "not established"),
    ],
)
def test_outcome_to_claim_table(args, fragment):
    assert fragment in protocol.claim(*args)


def test_paired_interval_levels():
    matrix = np.arange(12, dtype=float).reshape(3, 4) / 10
    out = protocol.paired_interval(matrix)
    assert out["n_seeds"] == 3 and out["n_tasks"] == 4
    assert out["ci95"][0] <= out["mean"] <= out["ci95"][1]
    with pytest.raises(ValueError):
        protocol.paired_interval(np.array([[np.nan, 0.0]]))


def test_reference_parity_with_the_package_loop():
    with jax.enable_x64(True):
        assert protocol.reference_parity_check()


def test_smoke_evaluation_and_decisions_on_training_tasks():
    with jax.enable_x64(True):
        specs = [s for s in protocol.training_specs()
                 if s["dimension"] == 4 and s["replicate"] < 2
                 and (s["family"], s["condition"]) in (("diagonal", 1000.0), ("rotated", 100.0),
                                                       ("rotated", 1000.0), ("rastrigin", 0.0))]
        learned = {arm: [(0, init_gram_model(jax.random.PRNGKey(1), protocol.ARMS[arm]))]
                   for arm in protocol.ARMS}
        rows = protocol.evaluate_rows(specs, learned, budgets=(100,))
        assert {r["arm"] for r in rows} == {"bb", "oracle", "lbfgsb", "tf-h8", *protocol.ARMS}
        assert all(r["units"] <= 100 for r in rows)
        stage1 = protocol.stage1_decision(rows)
        assert isinstance(stage1["S1_pass"], bool)
        cosines = protocol.probe_cosines(specs[:1], learned["gram-full"][0][1],
                                         protocol.ARMS["gram-full"], states=5)
        assert len(cosines) == 5 and all(-1 - 1e-9 <= c <= 1 + 1e-9 for c in cosines)
        protocol.GUARD = ("rastrigin",)  # the smoke tasks contain only this guard family
        try:
            stage2 = protocol.stage2_decision(rows, cosines)
        finally:
            protocol.GUARD = ("rastrigin", "ackley", "rosenbrock-shifted")
        assert isinstance(stage2["claim"], str) and "H2_context" in stage2
