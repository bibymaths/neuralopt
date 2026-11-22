"""Validation data for the amended C2 (continuous cost to success); harness only.

Per instance and per multistart start (the 16 Sobol starts of F0, seed 3), units to
success within 800 units for owned L-BFGS (A2o) and full-budget SciPy (A2-full). From them,
the sequential-restart multistart cost to success for a per-start cap p, which is the
continuous form of F0's "K starts with B/K units each". With A0 and A3 from
``results/f0-instances-v0.8.0.json``, this gives the power check for the amended C2
(``docs/research/v0.8-preregistration-amendments.md``, amendment 1).

Usage: run --family NAME --out RUN_DIR; combine --runs DIR... --output FILE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                      "intra_op_parallelism_threads=1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import f0_instances_v0_8 as fi  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import run_checkpoint  # noqa: E402
import run_f0_v0_8 as f0  # noqa: E402

from neuralopt.families import FAMILIES, make_split  # noqa: E402

INSTANCES = ROOT / "results" / "f0-instances-v0.8.0.json"
RUN_FILES = (ROOT / "scripts" / "c2_validation_v0_8.py", ROOT / "scripts" / "f0_instances_v0_8.py",
             ROOT / "scripts" / "run_f0_v0_8.py", ROOT / "src" / "neuralopt" / "families.py",
             ROOT / "src" / "neuralopt" / "lbfgs.py", ROOT / "uv.lock")


def extract(name: str, progress: Any) -> dict[str, Any]:
    spec = FAMILIES[name]
    n = f0.N_VALIDATION
    row = next(r for r in json.loads(INSTANCES.read_text())["families"] if r["family"] == name)
    f_star = np.asarray(row["per_instance"]["f_star"])
    val, _ = make_split(name, "validation", n)
    ctx = [jax.tree_util.tree_map(lambda a, i=i: a[i], val) for i in range(n)]
    starts = f0._sobol(spec, max(f0.MULTISTART_K), f0.SOBOL_SEEDS["arms"])
    t0 = time.perf_counter()
    owned, scipy_full = [], []
    vg = jax.jit(jax.value_and_grad(f0.family_objective(name)))
    for k in range(len(starts)):
        progress("starts", k, len(starts))
        traces = f0._owned_traces(name, val, np.tile(starts[k], (n, 1)), f0.LONG)
        owned.append([fi._units_to_success(t["trace"], f_star[i]) for i, t in enumerate(traces)])
        scipy_full.append([fi._units_to_success(fi.scipy_full_trace(vg, starts[k], ctx[i], spec,
                                                                    f0.LONG), f_star[i])
                           for i in range(n)])
    return {"family": name, "per_start_units_to_success": {"A2o": owned, "A2_full": scipy_full},
            "wall_seconds": time.perf_counter() - t0}


def run(name: str, run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    code = run_checkpoint.fingerprint(RUN_FILES)
    with (run_dir / "command.txt").open("a") as handle:
        handle.write(run_checkpoint.command_record(
            [sys.executable, *sys.argv],
            {"code_fingerprint": code,
             "cap": "1 CPU-hour for this process (process CPU clock; one pinned core); "
                    "2 CPU-hours for the three families"}))
    run_checkpoint.run_units([name], run_dir, extract, fingerprint=code)
    (run_dir / "DONE").write_text("units/\n")
    return 0


def combine(dirs: list[Path], output: Path) -> int:
    if output.exists():
        raise SystemExit(f"{output} exists; results are never overwritten")
    rows = [json.loads(p.read_text())["row"]
            for d in dirs for p in sorted((d / "units").glob("*.json"))]
    run_checkpoint.write_atomic(output, json.dumps({"schema": "neuralopt-c2-validation/v0.8",
                                                    "families": rows}) + "\n")
    print(f"wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=list(FAMILIES), required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("combine")
    c.add_argument("--runs", type=Path, nargs="+", required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.family, args.out) if args.command == "run" else combine(args.runs, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
