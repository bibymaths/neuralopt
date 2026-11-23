"""Main-track training summary (validation only): success by budget from units to success,
and the compute accounting with an explicit bound for stages that did not record CPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from training_summary_v0_8 import compute_accounting, success_by_budget  # noqa: E402


def test_success_by_budget_counts_censored_instances_as_failures():
    units = [10, 50, None, 200, 801]
    rates = success_by_budget(units, (25, 50, 200, 800))
    assert rates == {"25": 0.2, "50": 0.4, "200": 0.6, "800": 0.6}


def test_accounting_sums_recorded_cpu_and_bounds_unrecorded_stages_by_wall():
    stages = [
        {"family": "a", "stage": "R-seed0", "cpu_seconds": 3600.0, "wall_seconds": 4000.0},
        {"family": "a", "stage": "labels", "cpu_seconds": None, "wall_seconds": 1800.0},
    ]
    acc = compute_accounting(stages, cap_cpu_hours=1.2)
    assert acc["recorded_cpu_hours"] == pytest.approx(1.0)
    assert acc["unrecorded_wall_bound_hours"] == pytest.approx(0.5)
    assert acc["total_cpu_hours_upper_bound"] == pytest.approx(1.5)
    assert acc["over_cap"] is True
    assert acc["overrun_cpu_hours_range"] == pytest.approx([0.0, 0.3])
