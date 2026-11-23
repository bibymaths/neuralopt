"""Cap monitor for the v0.8 evaluation (CLAUDE.md rule 10).

Every ``--interval`` seconds it projects the evaluation's total process CPU:
  spent = CPU already spent before the run (validation checks) + each shard process's CPU;
  remaining = each shard's unfinished instances × its family's observed CPU per instance
              (from finished chunks; before any chunk finishes, the measured rate);
  projection = spent + remaining + overhead (timing stage and analysis).
If the projection exceeds the cap, it writes ``STOP_CAP``, interrupts every shard
(finished chunks are kept), and exits 3, so the work stops for the user's decision.

Usage: python scripts/eval_monitor_v0_8.py RUN_DIR --cap 24 --spent-before S --overhead S
       --rate FAMILY=SECONDS ...
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_checkpoint  # noqa: E402


def project(shards: list[dict[str, Any]], fallback_rate: dict[str, float], spent_before: float,
            overhead: float) -> dict[str, Any]:
    rates = {}
    for fam in {s["family"] for s in shards}:
        done = sum(s["done_instances"] for s in shards if s["family"] == fam)
        cpu = sum(s["done_chunk_cpu"] for s in shards if s["family"] == fam)
        rates[fam] = cpu / done if done else fallback_rate[fam]
    spent = spent_before + sum(s["process_cpu"] for s in shards)
    remaining = sum((s["instances"] - s["done_instances"]) * rates[s["family"]] for s in shards)
    return {"rate_seconds_per_instance": rates, "spent_cpu_seconds": spent,
            "remaining_cpu_seconds": remaining,
            "projected_cpu_hours": (spent + remaining + overhead) / 3600}


def _pids(out_dir: Path) -> list[int]:
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        text = [c.decode(errors="replace") for c in cmd]
        if any(t.endswith("python") or "python3" in t for t in text[:1]) and \
                "scripts/evaluate_v0_8.py" in text and str(out_dir) in text:
            found.append(int(p.name))
    return found


def _cpu_seconds(pid: int) -> float:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return 0.0
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def shard_states(run_dir: Path, cache: dict[str, float]) -> list[dict[str, Any]]:
    states = []
    for line in (run_dir / "shards.txt").read_text().split("\n"):
        if not line.strip():
            continue
        fam, split, start, stop = line.split()
        d = run_dir / "shards" / f"{fam}-{split}-{start}"
        rows = cpu = 0
        for p in sorted((d / "units").glob("*.json")) if (d / "units").exists() else []:
            row = json.loads(p.read_text())["row"]
            rows += len(row["rows"])
            cpu += row["cpu_seconds"]
        final = d / "process_cpu.json"
        if final.exists():
            proc = json.loads(final.read_text())["process_cpu_seconds"]
        else:
            live = sum(_cpu_seconds(pid) for pid in _pids(d))
            proc = max(live, cache.get(str(d), 0.0))
        cache[str(d)] = proc
        states.append({"family": fam, "dir": str(d), "instances": int(stop) - int(start),
                       "done_instances": rows, "done_chunk_cpu": cpu, "process_cpu": proc})
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--spent-before", type=float, required=True)
    parser.add_argument("--overhead", type=float, required=True)
    parser.add_argument("--rate", nargs="+", required=True)
    parser.add_argument("--interval", type=float, default=120.0)
    args = parser.parse_args()
    fallback = {k: float(v) for k, v in (r.split("=") for r in args.rate)}
    cache: dict[str, float] = {}
    while True:
        states = shard_states(args.run_dir, cache)
        out = project(states, fallback, args.spent_before, args.overhead)
        out.update(updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   cap_cpu_hours=args.cap, shards=states)
        run_checkpoint.write_atomic(args.run_dir / "monitor.json", json.dumps(out, indent=1))
        if out["projected_cpu_hours"] > args.cap:
            (args.run_dir / "STOP_CAP").write_text(json.dumps(out, indent=1))
            for s in states:
                for pid in _pids(Path(s["dir"])):
                    os.kill(pid, signal.SIGINT)
            print(f"STOP: projection {out['projected_cpu_hours']:.2f} CPU-h exceeds the cap "
                  f"{args.cap}", flush=True)
            return 3
        if all(s["done_instances"] == s["instances"] for s in states) or \
                (args.run_dir / "STOPPED").exists():
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
