"""Legacy checkpoints and default inference must be unchanged by v0.6 work."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from legacy_cases import legacy_outputs

GOLDEN = Path(__file__).parent / "data" / "legacy-v0.5-golden.json"

# The golden was generated locally; float32 results differ slightly across CPUs and
# XLA codegen. Worst drift measured on the CI Python 3.12 runner (run 44958858680):
#   stochastic cases: langevin_fixed rel 2.29e-5 (abs 3.81e-6 at x=0.167),
#     underdamped_learned rel 8.57e-6, langevin_learned rel 5.19e-6, HMC exact;
#   deterministic cases: direct_default rel 1.55e-7; checkpoint outputs abs <= 1.94e-7.
# Tolerances give roughly 4-7x margin; behavioural changes move these values by
# 1e-2 or more and are still rejected.
STOCHASTIC_CASES = ("langevin_learned", "underdamped_learned", "langevin_fixed",
                    "hamiltonian_identity")


def _tolerance(path: str) -> tuple[float, float]:
    """Return ``(rtol, atol)`` for a golden path such as ``/langevin_learned/x``."""
    return (1e-4, 1e-5) if path.split("/")[1] in STOCHASTIC_CASES else (1e-6, 1e-6)


def _collect(
    actual: object, expected: object, path: str, failures: list[str], drift: dict[str, list]
) -> None:
    """Walk the golden tree, recording every mismatch and the per-case maximum drift."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            failures.append(f"{path}: keys differ")
            return
        for key in expected:
            _collect(actual[key], expected[key], f"{path}/{key}", failures, drift)
    elif isinstance(expected, list) and not (
        expected and isinstance(expected[0], (int, float))
    ):
        if not isinstance(actual, list) or len(actual) != len(expected):
            failures.append(f"{path}: length differs")
            return
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _collect(a, e, f"{path}[{i}]", failures, drift)
    elif isinstance(expected, (list, float)):
        a = np.asarray(actual, dtype=float)
        e = np.asarray(expected, dtype=float)
        if a.shape != e.shape:
            failures.append(f"{path}: shape {a.shape} != {e.shape}")
            return
        rtol, atol = _tolerance(path)
        error = np.abs(a - e)
        rel = error / np.maximum(np.abs(e), 1e-30)
        case = drift.setdefault(path.split("/")[1], [0.0, 0.0])
        case[0] = max(case[0], float(error.max(initial=0.0)))
        case[1] = max(case[1], float(rel.max(initial=0.0)))
        if np.any(error > atol + rtol * np.abs(e)):
            failures.append(
                f"{path}: max_abs={error.max():.2e} max_rel={rel.max():.2e} "
                f"(rtol={rtol:g}, atol={atol:g})"
            )
    elif actual != expected:
        failures.append(f"{path}: {actual!r} != {expected!r}")


def _assert_matches(actual: object, expected: object) -> None:
    """Fail with every mismatch, plus the per-case drift summary for passing cases too."""
    failures: list[str] = []
    drift: dict[str, list] = {}
    _collect(actual, expected, "", failures, drift)
    if failures:
        summary = "\n".join(
            f"  {case}: max_abs={a:.2e} max_rel={r:.2e}" for case, (a, r) in sorted(drift.items())
        )
        raise AssertionError(
            f"{len(failures)} golden mismatch(es):\n  " + "\n  ".join(failures)
            + f"\nper-case maximum drift:\n{summary}"
        )


def test_legacy_outputs_match_frozen_v05_golden():
    expected = json.loads(GOLDEN.read_text())
    _assert_matches(legacy_outputs(), expected)


def test_tolerance_policy_separates_stochastic_from_exact_cases():
    for case in STOCHASTIC_CASES:
        assert _tolerance(f"/{case}/trajectory") == (1e-4, 1e-5)
    assert _tolerance("/direct_default/trajectory") == (1e-6, 1e-6)
    assert _tolerance("/checkpoint:demo-v0.1.0.npz/outputs/diag") == (1e-6, 1e-6)


def test_stochastic_tolerance_absorbs_host_drift_but_catches_real_changes():
    golden = {
        "langevin_learned": {"trajectory": [152.922867, 83.7913589477539, 33.263927]},
        "langevin_fixed": {"x": [1.761203, 0.16666126251220703]},
    }
    # Worst cross-host float32 drift measured in CI (Python 3.12 runner).
    _assert_matches(
        {
            "langevin_learned": {"trajectory": [152.922867, 83.79119873046875, 33.263916]},
            "langevin_fixed": {"x": [1.761203, 0.1666574478149414]},
        },
        golden,
    )
    # A behavioural change (here 2.5e-3 relative) is still rejected.
    with pytest.raises(AssertionError):
        _assert_matches(
            {
                "langevin_learned": {"trajectory": [152.922867, 84.0, 33.263927]},
                "langevin_fixed": {"x": [1.761203, 0.16666126251220703]},
            },
            golden,
        )
    with pytest.raises(AssertionError):
        _assert_matches({"direct_default": {"fun": 1.00001}}, {"direct_default": {"fun": 1.0}})


def test_every_mismatch_is_reported_with_its_size():
    golden = {
        "langevin_learned": {"trajectory": [1.0, 2.0]},
        "langevin_fixed": {"x": [0.5, -0.25], "fun": 3.0},
        "checkpoint:a.npz": {"config": [16, 4]},
    }
    actual = {
        "langevin_learned": {"trajectory": [1.0, 2.1]},
        "langevin_fixed": {"x": [0.5, -0.2], "fun": 3.0},
        "checkpoint:a.npz": {"config": [16, 4]},
    }
    with pytest.raises(AssertionError) as info:
        _assert_matches(actual, golden)
    message = str(info.value)
    assert "/langevin_learned/trajectory" in message and "/langevin_fixed/x" in message
    assert "max_rel=5.00e-02" in message and "max_abs=5.00e-02" in message
    # Per-case drift summary covers passing cases too.
    assert "checkpoint:a.npz" in message
