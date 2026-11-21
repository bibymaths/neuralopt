"""Legacy checkpoints and default inference must be unchanged by v0.6 work."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from legacy_cases import legacy_outputs

GOLDEN = Path(__file__).parent / "data" / "legacy-v0.5-golden.json"


def _compare(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            _compare(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list) and expected and isinstance(expected[0], (int, float)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, err_msg=path)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, err_msg=path)
    else:
        assert actual == expected, path


def test_legacy_outputs_match_frozen_v05_golden():
    expected = json.loads(GOLDEN.read_text())
    _compare(legacy_outputs(), expected, "")
