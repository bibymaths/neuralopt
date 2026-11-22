"""Run-directory checkpointing and resume for long harness runs (scripts/run_checkpoint.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("run_checkpoint",
                                                  ROOT / "scripts" / "run_checkpoint.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_checkpoint"] = module
    spec.loader.exec_module(module)
    return module


class _Recorder:
    def __init__(self, fail_on: str | None = None):
        self.calls: list[str] = []
        self.fail_on = fail_on

    def __call__(self, name, progress):
        self.calls.append(name)
        if name == self.fail_on:
            raise RuntimeError("interrupted")
        for i in range(3):
            progress("stage", i + 1, 3)
        return {"family": name, "value": len(name)}


def test_each_finished_unit_is_checkpointed_and_progress_is_reported(tmp_path):
    rc = _module()
    compute = _Recorder()
    rows = rc.run_units(["a", "bb", "ccc"], tmp_path, compute, fingerprint="f1",
                        weights={"a": 1.0, "bb": 1.0, "ccc": 2.0})
    assert [r["family"] for r in rows] == ["a", "bb", "ccc"]
    assert compute.calls == ["a", "bb", "ccc"]
    for name in ("a", "bb", "ccc"):
        saved = json.loads((tmp_path / "units" / f"{name}.json").read_text())
        assert saved["fingerprint"] == "f1" and saved["row"]["family"] == name
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["units_done"] == 3 and progress["units_total"] == 3
    assert progress["current"] is None and progress["eta_seconds"] == 0.0
    assert "elapsed_seconds" in progress and "updated_utc" in progress
    assert not list(tmp_path.glob("**/*.tmp"))  # atomic writes leave no temporaries


def test_resume_skips_finished_units_and_leaves_their_checkpoints_untouched(tmp_path):
    rc = _module()
    with pytest.raises(RuntimeError):
        rc.run_units(["a", "bb", "ccc"], tmp_path, _Recorder(fail_on="bb"), fingerprint="f1")
    kept = (tmp_path / "units" / "a.json").read_bytes()
    assert not (tmp_path / "units" / "bb.json").exists()

    compute = _Recorder()
    rows = rc.run_units(["a", "bb", "ccc"], tmp_path, compute, fingerprint="f1")
    assert compute.calls == ["bb", "ccc"]  # finished work is not redone
    assert (tmp_path / "units" / "a.json").read_bytes() == kept
    assert [r["family"] for r in rows] == ["a", "bb", "ccc"]


def test_resume_refuses_checkpoints_from_different_code(tmp_path):
    rc = _module()
    rc.run_units(["a"], tmp_path, _Recorder(), fingerprint="f1")
    with pytest.raises(SystemExit, match="fingerprint"):
        rc.run_units(["a", "bb"], tmp_path, _Recorder(), fingerprint="f2")


def test_progress_reports_the_current_stage_while_a_unit_runs(tmp_path):
    rc = _module()
    seen = []

    def compute(name, progress):
        progress("reference", 2, 5)
        seen.append(json.loads((tmp_path / "progress.json").read_text()))
        return {"family": name}

    rc.run_units(["a"], tmp_path, compute, fingerprint="f1", min_interval=0.0)
    assert seen[0]["current"] == {"unit": "a", "stage": "reference",
                                  "instances_done": 2, "instances_total": 5}
    assert seen[0]["units_done"] == 0


def test_fingerprint_changes_with_file_contents(tmp_path):
    rc = _module()
    f = tmp_path / "x.py"
    f.write_text("a = 1\n")
    first = rc.fingerprint([f])
    f.write_text("a = 2\n")
    assert rc.fingerprint([f]) != first
