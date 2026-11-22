"""Frozen v0.8 pre-registrations must not change (amendments go in separate files)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stem", ["v0.8-preregistration", "v0.8-b-preregistration"])
def test_frozen_preregistration_matches_its_recorded_hash(stem):
    recorded, name = (ROOT / f"docs/research/{stem}.sha256").read_text().split()
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert actual == recorded
    assert recorded in (ROOT / "docs/research/v0.8-claims.md").read_text()
