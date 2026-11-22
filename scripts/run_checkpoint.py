"""Run-directory checkpointing for long harness runs.

A run directory holds one checkpoint per finished unit (for F0: one per family) under
``units/<name>.json`` and a ``progress.json`` that is rewritten atomically. Re-running with the
same directory resumes: finished units are loaded, never recomputed or rewritten. Checkpoints
carry a fingerprint of the code that produced them; a mismatch refuses to resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Progress = Callable[[str, int, int], None]


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).name.encode() + b"\0" + Path(path).read_bytes() + b"\0")
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")


def _git(*args: str) -> str:
    import subprocess

    root = Path(__file__).resolve().parents[1]
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()


def command_record(argv: Sequence[str], extra: dict[str, str] | None = None) -> str:
    """Provenance block for ``command.txt``: command, git state, uv.lock hash, start time,
    thread settings and CPU affinity (CLAUDE.md rule 9), plus ``extra`` (e.g. the cap)."""
    root = Path(__file__).resolve().parents[1]
    lock = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not on Linux
        affinity = []
    lines = [f"start_utc: {_now()}", f"command: {' '.join(argv)}",
             f"git_sha: {_git('rev-parse', 'HEAD')}",
             f"git_dirty: {bool(_git('status', '--porcelain', '--untracked-files=no'))}",
             f"uv_lock_sha256: {lock}"]
    lines += [f"{k}={os.environ.get(k, '')}" for k in THREAD_VARIABLES]
    lines.append(f"cpu_affinity: {affinity}")
    lines += [f"{k}: {v}" for k, v in (extra or {}).items()]
    return "\n".join(lines) + "\n\n"


def run_units(names: Sequence[str], run_dir: Path, compute: Callable[[str, Progress], Any], *,
              fingerprint: str, weights: dict[str, float] | None = None,
              min_interval: float = 30.0) -> list[Any]:
    """Run ``compute(name, progress)`` for each unit not yet checkpointed in ``run_dir``.

    ``weights`` (for example projected hours per unit) shape the ETA; equal weights otherwise.
    """
    run_dir = Path(run_dir)
    units = run_dir / "units"
    weights = weights or {}
    weight = {n: float(weights.get(n, 1.0)) for n in names}
    rows: dict[str, Any] = {}
    for name in names:
        path = units / f"{name}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("fingerprint") != fingerprint:
                raise SystemExit(f"{path}: fingerprint {saved.get('fingerprint')} does not match "
                                 f"the current code ({fingerprint}); refusing to resume")
            rows[name] = saved["row"]

    t0 = time.monotonic()
    computed_weight = 0.0  # weight of units computed in this process (for the ETA rate)
    state: dict[str, Any] = {"current": None, "last": -float("inf")}

    def report(force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - state["last"] < min_interval:
            return
        state["last"] = now
        elapsed = now - t0
        remaining = sum(weight[n] for n in names if n not in rows)
        if remaining == 0:
            eta: float | None = 0.0
        elif computed_weight > 0:
            eta = remaining * elapsed / computed_weight
        else:
            eta = None
        write_atomic(run_dir / "progress.json", json.dumps({
            "updated_utc": _now(),
            "units_done": len(rows), "units_total": len(names),
            "done": [n for n in names if n in rows],
            "current": state["current"],
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": None if eta is None else round(eta, 1),
        }, indent=1) + "\n")

    report(force=True)
    for name in names:
        if name in rows:
            continue

        def progress(stage: str, done: int, total: int, name: str = name) -> None:
            new_stage = state["current"] is None or state["current"].get("stage") != stage
            state["current"] = {"unit": name, "stage": stage,
                                "instances_done": int(done), "instances_total": int(total)}
            report(force=new_stage or done == total)

        state["current"] = {"unit": name, "stage": "start", "instances_done": 0,
                            "instances_total": 0}
        report(force=True)
        row = compute(name, progress)
        write_atomic(units / f"{name}.json",
                     json.dumps({"fingerprint": fingerprint, "finished_utc": _now(),
                                 "row": row}, indent=2) + "\n")
        rows[name] = row
        computed_weight += weight[name]
        state["current"] = None
        report(force=True)
    return [rows[n] for n in names]
