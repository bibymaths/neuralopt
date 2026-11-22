"""Per-instance F0 data for the v0.8 pre-registration's power check (harness only).

Recomputes gate F0 on the validation split exactly as ``run_f0_v0_8.py`` does, imported
unchanged. It keeps per-instance outcomes: units to success for the single-start arms, and
success per (K, budget) for the multistart arms. It also adds the pre-registered SciPy
setting, with tolerances that let SciPy use the full budget (ftol = gtol = 0,
maxfun = budget/2), as arms A1-full and A2-full.

It must reproduce the frozen F0 aggregates (``results/f0-v0.8.0.json``) exactly; the
comparison is stored in the output.

Usage: run --family NAME --out RUN_DIR      (one family per process; long-run rule)
       combine --runs DIR... --output FILE  (writes the versioned results file)
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
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import run_checkpoint  # noqa: E402
import run_f0_v0_8 as f0  # noqa: E402  (sets float64)

from neuralopt.families import FAMILIES, make_split  # noqa: E402

RUN_FILES = (ROOT / "scripts" / "f0_instances_v0_8.py", ROOT / "scripts" / "run_f0_v0_8.py",
             ROOT / "scripts" / "run_checkpoint.py", ROOT / "src" / "neuralopt" / "families.py",
             ROOT / "src" / "neuralopt" / "lbfgs.py", ROOT / "uv.lock")
FROZEN = ROOT / "results" / "f0-v0.8.0.json"


def scipy_full_trace(vg: Any, x0: np.ndarray, c: Any, spec: Any, max_units: int
                     ) -> list[tuple[int, float]]:
    """SciPy L-BFGS-B with ftol = gtol = 0 and maxfun = budget/2 (2 units per call); the
    trace convention of ``run_f0_v0_8._ScipyRunner`` (best-so-far after each call)."""
    from scipy.optimize import minimize as scipy_minimize

    trace: list[tuple[int, float]] = []

    class Exhausted(Exception):
        pass

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        if 2 * (len(trace) + 1) > max_units:
            raise Exhausted
        v, g = vg(jnp.asarray(x), c)
        v = float(v)
        v = v if np.isfinite(v) else np.inf
        best = min(v, trace[-1][1] if trace else np.inf)
        trace.append((2 * (len(trace) + 1), best))
        g = np.asarray(g, float)
        return v, np.where(np.isfinite(g), g, 0.0)

    try:
        scipy_minimize(fun, np.clip(x0, spec.lo, spec.hi), jac=True, method="L-BFGS-B",
                       bounds=list(zip(spec.lo, spec.hi, strict=True)),
                       options={"maxfun": max_units // 2, "maxiter": max_units, "ftol": 0.0,
                                "gtol": 0.0, "maxcor": 10})
    except Exhausted:
        pass
    return trace


def _units_to_success(trace: list[tuple[int, float]], f_star: float) -> int | None:
    for u, v in trace:
        if f0._success(v, f_star):
            return int(u)
    return None


def extract(name: str, progress: Any) -> dict[str, Any]:
    spec = FAMILIES[name]
    n = f0.N_VALIDATION
    val, val_truth = make_split(name, "validation", n)
    ctx = [jax.tree_util.tree_map(lambda a, i=i: a[i], val) for i in range(n)]
    scipy = f0._ScipyRunner(name)
    t0 = time.perf_counter()

    ref = {}
    for tag in ("reference_a", "reference_b"):
        starts = f0._sobol(spec, f0.REFERENCE_STARTS, f0.SOBOL_SEEDS[tag])
        best = np.full(n, np.inf)
        for i in range(n):
            progress(tag, i, n)
            finals = [tr[-1][1] for x0 in [*starts, np.asarray(val_truth[i])]
                      if (tr := scipy.trace(x0, ctx[i], f0.REFERENCE_UNITS, precise=True))]
            best[i] = min(finals)
        ref[tag] = best
    f_star = np.minimum(ref["reference_a"], ref["reference_b"])

    progress("arms", 0, 1)
    default = np.tile(spec.default, (n, 1))
    a0 = f0._owned_traces(name, val, default, f0.LONG)
    a1 = [scipy.trace(spec.default, ctx[i], f0.LONG) for i in range(n)]
    arm_starts = f0._sobol(spec, max(f0.MULTISTART_K), f0.SOBOL_SEEDS["arms"])
    a2 = [[scipy.trace(arm_starts[k], ctx[i], f0.LONG) for k in range(len(arm_starts))]
          for i in range(n)]
    a2o = [f0._owned_traces(name, val, np.tile(arm_starts[k], (n, 1)), f0.LONG)
           for k in range(len(arm_starts))]
    vg = jax.jit(jax.value_and_grad(f0.family_objective(name)))
    progress("full-budget SciPy", 0, n)
    a1_full = [scipy_full_trace(vg, spec.default, ctx[i], spec, f0.LONG) for i in range(n)]
    a2_full = [[scipy_full_trace(vg, arm_starts[k], ctx[i], spec, f0.LONG)
                for k in range(len(arm_starts))] for i in range(n)]

    progress("labels and nearest neighbour", 0, 1)
    train, _ = make_split(name, "train", f0.N_TRAIN)
    label_starts = f0._sobol(spec, f0.LABEL_STARTS, f0.SOBOL_SEEDS["arms"] + 10)
    best_val = np.full(f0.N_TRAIN, np.inf)
    best_x = np.zeros((f0.N_TRAIN, spec.dimension))
    for k in range(f0.LABEL_STARTS):
        for i, r in enumerate(f0._owned_traces(name, train, np.tile(label_starts[k],
                                                                    (f0.N_TRAIN, 1)),
                                               f0.LABEL_UNITS)):
            if r["trace"][-1][1] < best_val[i]:
                best_val[i], best_x[i] = r["trace"][-1][1], r["x"]
    train_feats = f0._nn_features(name, train)
    mean, std = train_feats.mean(0), train_feats.std(0) + 1e-12
    z_train = (train_feats - mean) / std
    z_val = (f0._nn_features(name, val) - mean) / std
    nearest = np.argmin(((z_val[:, None, :] - z_train[None, :, :]) ** 2).sum(-1), axis=1)
    a3 = f0._owned_traces(name, val, best_x[nearest], f0.LONG)

    # F0's f* update: any arm's final value below f* (A1-full/A2-full also count here, but
    # the F0 reproduction check below uses F0's own arm set)
    arm_best_f0 = np.array([min([a0[i]["trace"][-1][1], a1[i][-1][1] if a1[i] else np.inf,
                                 a3[i]["trace"][-1][1], *[t[-1][1] for t in a2[i] if t],
                                 *[a2o[k][i]["trace"][-1][1] for k in range(len(arm_starts))]])
                            for i in range(n)])
    f_star_f0 = np.minimum(f_star, arm_best_f0)
    full_best = np.array([min([a1_full[i][-1][1] if a1_full[i] else np.inf,
                               *[t[-1][1] for t in a2_full[i] if t]]) for i in range(n)])
    f_star_all = np.minimum(f_star_f0, full_best)

    def single(traces: list[Any], key: str | None, fs: np.ndarray) -> dict[str, Any]:
        tr = [t[key] if key else t for t in traces]
        return {"units_to_success": [_units_to_success(t, fs[i]) for i, t in enumerate(tr)],
                "final": [t[-1][1] if t else None for t in tr]}

    def multi(get: Any, fs: np.ndarray) -> dict[str, Any]:
        out = {}
        for k in f0.MULTISTART_K:
            out[str(k)] = {}
            for b in f0.BUDGETS:
                per = b // k
                if per < 2:
                    continue
                out[str(k)][str(b)] = [bool(f0._success(min(f0._best_within(get(i, s), per)
                                                             for s in range(k)), fs[i]))
                                       for i in range(n)]
        return out

    per_instance = {
        "f_star": f_star_f0.tolist(), "f_star_with_full_budget_scipy": f_star_all.tolist(),
        "A0": single(a0, "trace", f_star_f0), "A1": single(a1, None, f_star_f0),
        "A1_full": single(a1_full, None, f_star_f0), "A3": single(a3, "trace", f_star_f0),
        "A0_used": [t["used"] for t in a0], "A3_used": [t["used"] for t in a3],
        "A2": multi(lambda i, s: a2[i][s], f_star_f0),
        "A2_full": multi(lambda i, s: a2_full[i][s], f_star_f0),
        "A2o": multi(lambda i, s: a2o[s][i]["trace"], f_star_f0),
        "nearest_train_index": nearest.tolist(),
        "truth": np.asarray(val_truth).tolist(),
        "A3_start": best_x[nearest].tolist(),
    }

    # reproduce F0's aggregate success rates exactly
    frozen = next(r for r in json.loads(FROZEN.read_text())["families"] if r["family"] == name)
    rates = frozen["success_rate_by_budget"]

    def rate_single(arm: str) -> dict[str, float]:
        u = per_instance[arm]["units_to_success"]
        return {str(b): float(np.mean([x is not None and x <= b for x in u])) for b in f0.BUDGETS}

    def rate_multi(arm: str) -> dict[str, float]:
        out = {}
        for b in f0.BUDGETS:
            best = -1.0
            for k in f0.MULTISTART_K:
                if str(b) in per_instance[arm][str(k)]:
                    best = max(best, float(np.mean(per_instance[arm][str(k)][str(b)])))
            out[str(b)] = best
        return out

    mine = {"A0_default_owned": rate_single("A0"), "A1_default_scipy": rate_single("A1"),
            "A3_nearest_neighbour_owned": rate_single("A3"),
            "A2_multistart_scipy": rate_multi("A2"), "A2o_multistart_owned": rate_multi("A2o")}
    reproduced = {arm: mine[arm] == {str(k): v for k, v in rates[arm].items()} for arm in mine}
    return {"family": name, "per_instance": per_instance, "recomputed_rates": mine,
            "full_budget_scipy_rates": {"A1_full": rate_single("A1_full"),
                                        "A2_full": rate_multi("A2_full")},
            "reproduces_frozen_f0": reproduced,
            "f_star_lowered_by_full_budget_scipy": int(np.sum(
                full_best < f_star_f0 - (f0.EPS_ABS + f0.EPS_REL * np.abs(f_star_f0)))),
            "wall_seconds": time.perf_counter() - t0}


def run(name: str, run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    code = run_checkpoint.fingerprint(RUN_FILES)
    with (run_dir / "command.txt").open("a") as handle:
        handle.write(run_checkpoint.command_record(
            [sys.executable, *sys.argv],
            {"code_fingerprint": code,
             "cap": "1.5 CPU-hours for this process (process CPU clock; one pinned core); "
                    "4 CPU-hours for the three families"}))
    rows = run_checkpoint.run_units([name], run_dir, extract, fingerprint=code)
    print(json.dumps({"family": name, "reproduces_frozen_f0": rows[0]["reproduces_frozen_f0"],
                      "full_budget_scipy_rates": rows[0]["full_budget_scipy_rates"]}),
          flush=True)
    (run_dir / "DONE").write_text("units/\n")
    return 0


def combine(dirs: list[Path], output: Path) -> int:
    if output.exists():
        raise SystemExit(f"{output} exists; results are never overwritten")
    rows = []
    for d in dirs:
        if not (d / "DONE").exists():
            raise SystemExit(f"{d} has not finished")
        rows.extend(json.loads(p.read_text())["row"] for p in sorted((d / "units").glob("*.json")))
    run_checkpoint.write_atomic(output, json.dumps(
        {"schema": "neuralopt-f0-instances/v0.8", "frozen_f0": str(FROZEN.relative_to(ROOT)),
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
