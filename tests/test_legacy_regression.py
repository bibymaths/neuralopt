"""Legacy checkpoints and default inference must be unchanged by v0.6 work."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from legacy_cases import legacy_outputs

GOLDEN = Path(__file__).parent / "data" / "legacy-v0.5-golden.json"

# Stochastic transitions accumulate float32 rounding that differs across CPUs/XLA
# codegen (CI observed 1.9e-6 relative drift against the locally generated golden).
# A behavioural change moves these values by orders of magnitude more than 1e-5.
# Deterministic direct inference, checkpoint outputs and configs stay at 1e-6.
STOCHASTIC_CASES = ("langevin_learned", "underdamped_learned", "langevin_fixed",
                    "hamiltonian_identity")


def _tolerance(path: str) -> tuple[float, float]:
    """Return ``(rtol, atol)`` for a golden path such as ``/langevin_learned/x``."""
    return (1e-5, 1e-6) if path.split("/")[1] in STOCHASTIC_CASES else (1e-6, 1e-7)


def _compare(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            _compare(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list) and expected and isinstance(expected[0], (int, float)):
        rtol, atol = _tolerance(path)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=path)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        rtol, atol = _tolerance(path)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=path)
    else:
        assert actual == expected, path


def test_legacy_outputs_match_frozen_v05_golden():
    expected = json.loads(GOLDEN.read_text())
    _compare(legacy_outputs(), expected, "")


def test_tolerance_policy_separates_stochastic_from_exact_cases():
    for case in STOCHASTIC_CASES:
        assert _tolerance(f"/{case}/trajectory") == (1e-5, 1e-6)
    assert _tolerance("/direct_default/trajectory") == (1e-6, 1e-7)
    assert _tolerance("/checkpoint:demo-v0.1.0.npz/outputs/diag") == (1e-6, 1e-7)


def test_stochastic_tolerance_absorbs_host_drift_but_catches_real_changes():
    golden = {"langevin_learned": {"trajectory": [152.922867, 83.7913589477539, 33.263927]}}
    # Cross-host float32 drift observed in CI (Python 3.12 runner), 1.9e-6 relative.
    _compare({"langevin_learned": {"trajectory": [152.922867, 83.79119873046875, 33.263916]}},
             golden, "")
    with pytest.raises(AssertionError):
        _compare({"langevin_learned": {"trajectory": [152.922867, 83.80, 33.263927]}},
                 golden, "")
    with pytest.raises(AssertionError):
        _compare({"direct_default": {"fun": 1.000002}}, {"direct_default": {"fun": 1.0}}, "")
