"""Amended-C2 helpers: sequential-restart cost to success and the per-instance area."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("power_c2_v0_8",
                                                  ROOT / "scripts" / "power_c2_v0_8.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["power_c2_v0_8"] = module
    spec.loader.exec_module(module)
    return module


def test_sequential_restarts_charge_full_caps_for_failed_starts():
    m = _module()
    assert m.sequential_cost([30, None], cap=100) == 30
    assert m.sequential_cost([150, 40], cap=100) == 140  # start 0 fails within its cap
    assert m.sequential_cost([None, None, 10], cap=100) == 210
    assert m.sequential_cost([None] * 8 + [5], cap=100) is None  # start 8 would begin at 800
    assert m.sequential_cost([None, 300], cap=400) == 700


def test_last_start_is_truncated_at_the_budget():
    m = _module()
    assert m.sequential_cost([None, 350], cap=500) is None  # only 300 units left for start 1
    assert m.sequential_cost([None, 250], cap=500) == 750


def test_area_is_the_share_of_grid_budgets_already_solved():
    m = _module()
    assert m.area(None) == 0.0 and m.area(801) == 0.0
    assert m.area(10) == 1.0 and m.area(800) == 1 / 80
    assert m.area(15) == 79 / 80
