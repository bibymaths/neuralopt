"""Main track: combine the training run's validation-only choices and compute accounting.

Reads ``runs/<train run>/<family>/{labels,R-seedS,O-seedS,validate}/`` and writes one
versioned result (never overwritten). Validation only: no test or OOD split is opened.

Stages that did not record process CPU (labels, validate) ran on one pinned core
(``taskset -c N``), so their process CPU is at most their wall time; the accounting reports
that bound, not a guess.

Usage: python scripts/training_summary_v0_8.py RUN_DIR --output FILE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FAMILIES = ("sinusoid", "cascade-clean", "cascade-realistic")
BUDGETS = (25, 50, 100, 200, 400, 800)
CAP_TRAINING_CPU_HOURS = 26.0  # amendment 3
CAP_PER_PROCESS_CPU_HOURS = 3.0  # amendment 3
F0 = ROOT / "results" / "f0-v0.8.0.json"


def success_by_budget(units: list[int | None], budgets: tuple[int, ...]) -> dict[str, float]:
    """Share of instances whose units to first success are within each budget."""
    n = len(units)
    return {str(b): sum(u is not None and u <= b for u in units) / n for b in budgets}


def compute_accounting(stages: list[dict[str, Any]], cap_cpu_hours: float) -> dict[str, Any]:
    recorded = sum(s["cpu_seconds"] for s in stages if s["cpu_seconds"] is not None) / 3600
    bound = sum(s["wall_seconds"] for s in stages if s["cpu_seconds"] is None) / 3600
    upper = recorded + bound
    return {"recorded_cpu_hours": recorded, "unrecorded_wall_bound_hours": bound,
            "total_cpu_hours_upper_bound": upper, "cap_cpu_hours": cap_cpu_hours,
            "over_cap": upper > cap_cpu_hours,
            "overrun_cpu_hours_range": [max(recorded - cap_cpu_hours, 0.0),
                                        max(upper - cap_cpu_hours, 0.0)],
            "stages": stages}


def _start_utc(stage_dir: Path) -> datetime:
    for line in (stage_dir / "command.txt").read_text().splitlines():
        if line.startswith("start_utc:"):
            return datetime.strptime(line.split(":", 1)[1].strip(),
                                     "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    raise SystemExit(f"no start_utc in {stage_dir / 'command.txt'}")


def _stage(family: str, stage_dir: Path) -> dict[str, Any]:
    if (stage_dir / "exit_code").read_text().strip() != "0" or not (stage_dir / "DONE").exists():
        raise SystemExit(f"{stage_dir} did not finish successfully")
    result = json.loads((stage_dir / "result.json").read_text())
    end = datetime.fromtimestamp(os.path.getmtime(stage_dir / "exit_code"), UTC)
    wall = (end - _start_utc(stage_dir)).total_seconds()
    return {"family": family, "stage": stage_dir.name, "cpu_seconds": result.get("cpu_seconds"),
            "wall_seconds": wall}


def summarize(run_dir: Path) -> dict[str, Any]:
    f0 = {f["family"]: f["success_rate_by_budget"] for f in json.loads(F0.read_text())["families"]}
    families, stages = [], []
    for name in FAMILIES:
        fam_dir = run_dir / name
        for d in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            stages.append(_stage(name, d))
        v = json.loads((fam_dir / "validate" / "result.json").read_text())
        per_run = {}
        for key, r in v["runs"].items():
            per_run[key] = {**r, "success_by_budget": success_by_budget(r["units_to_success"],
                                                                        BUDGETS)}
        chosen = [r for r in per_run.values() if r["loss"] == v["chosen_loss"]]
        mean_curve = {str(b): sum(r["success_by_budget"][str(b)] for r in chosen) / len(chosen)
                      for b in BUDGETS}
        families.append({
            **{k: v[k] for k in ("family", "matched_budget", "success_at_B_by_loss",
                                 "chosen_loss", "tau_fit", "tau_c",
                                 "validation_silent_miss_at_tau_c",
                                 "validation_fallback_rate_at_tau_c", "timing")},
            "chosen_loss_success_by_budget_mean_over_seeds": mean_curve,
            "f0_bars_same_validation_instances": f0[name],
            "runs": per_run})
    accounting = compute_accounting(stages, CAP_TRAINING_CPU_HOURS)
    per_process_max = max((s["cpu_seconds"] or s["wall_seconds"]) for s in stages) / 3600
    accounting["per_process_cap_cpu_hours"] = CAP_PER_PROCESS_CPU_HOURS
    accounting["per_process_max_cpu_hours_upper_bound"] = per_process_max
    return {"schema": "neuralopt-v0.8-validation-choices/v1",
            "split": "validation only (200 instances per family; the F0 validation instances)",
            "run": run_dir.name, "families": families, "compute": accounting}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; results are never overwritten")
    summary = summarize(args.run_dir)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=1) + "\n")
    tmp.replace(args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
