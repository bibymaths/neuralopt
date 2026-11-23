"""Cap monitor for the v0.8 evaluation (CLAUDE.md rule 10): the projection of total process
CPU from finished chunks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_monitor_v0_8 import project  # noqa: E402


def test_projection_uses_each_family_s_observed_rate_for_the_remaining_instances():
    shards = [
        {"family": "a", "instances": 100, "done_instances": 50, "done_chunk_cpu": 100.0,
         "process_cpu": 130.0},
        {"family": "a", "instances": 100, "done_instances": 0, "done_chunk_cpu": 0.0,
         "process_cpu": 10.0},
        {"family": "b", "instances": 10, "done_instances": 10, "done_chunk_cpu": 50.0,
         "process_cpu": 55.0},
    ]
    out = project(shards, fallback_rate={"a": 99.0, "b": 99.0}, spent_before=3600.0,
                  overhead=1800.0)
    # family a rate 100/50 = 2 s per instance; 150 instances remain
    assert out["rate_seconds_per_instance"]["a"] == pytest.approx(2.0)
    assert out["remaining_cpu_seconds"] == pytest.approx(300.0)
    assert out["spent_cpu_seconds"] == pytest.approx(3600.0 + 130.0 + 10.0 + 55.0)
    assert out["projected_cpu_hours"] == pytest.approx((3600 + 195 + 300 + 1800) / 3600)


def test_projection_falls_back_to_the_measured_rate_before_any_chunk_finishes():
    shards = [{"family": "a", "instances": 25, "done_instances": 0, "done_chunk_cpu": 0.0,
               "process_cpu": 0.0}]
    out = project(shards, fallback_rate={"a": 4.0}, spent_before=0.0, overhead=0.0)
    assert out["remaining_cpu_seconds"] == pytest.approx(100.0)
