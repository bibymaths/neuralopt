"""The frozen v0.8 pre-registration must not change (amendments go in a separate file)."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_preregistration_matches_its_recorded_hash():
    recorded, name = (ROOT / "docs/research/v0.8-preregistration.sha256").read_text().split()
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert actual == recorded
    assert recorded in (ROOT / "docs/research/v0.8-claims.md").read_text()
