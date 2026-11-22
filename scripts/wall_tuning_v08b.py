"""v0.8-B: tune the baselines at their wall-matched budgets on validation (harness only).

For each tested family and policy budget B in {20, 50, 100}, the wall budget is
``T(B) = s_policy · B`` (the policy's frozen warm seconds per unit, from
``results/v0.8-b-timing.json``). Baseline k gets ``B_k = max{t : C_k(t) <= T(B)}`` units,
with C_k its frozen cumulative-time curve: linear with slope C_k(100)/100, except TPE, which
uses the measured curve and its slope over units 51-100 beyond 100. Each configuration of
the B-S2 grid runs at B_k on validation. The wall-matched anytime regret is
``(1/T) ∫_0^T L(τ) dτ``, with L the clipped log regret of the best value available by time τ
(mapping units to time through C_k). The basin rate uses the polish from the best point
within B_k units.

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
import numpy as np  # noqa: E402
import run_b0_v0_8b as b0  # noqa: E402  (sets float64)
import run_checkpoint  # noqa: E402
import v08b_baselines as bl  # noqa: E402

TIMING = ROOT / "results" / "v0.8-b-timing.json"
B0_RESULTS = ROOT / "results" / "v0.8-b-b0.json"
RUN_FILES = (ROOT / "scripts" / "wall_tuning_v08b.py", ROOT / "scripts" / "run_b0_v0_8b.py",
             ROOT / "scripts" / "v08b_baselines.py", TIMING, B0_RESULTS, ROOT / "uv.lock")


def curves(name: str) -> tuple[float, dict[str, Any]]:
    fam = next(f for f in json.loads(TIMING.read_text())["families"] if f["family"] == name)
    s_pol = fam["policy"]["rollout_seconds_per_instance_median"] / 100.0
    out = {}
    for arm, v in fam["baselines"].items():
        c = np.asarray(v["cumulative_seconds"], float)
        if arm == "tpe":
            slope = (c[99] - c[49]) / 50.0
            out[arm] = (lambda t, c=c, slope=slope:
                        0.0 if t <= 0 else (c[t - 1] if t <= 100 else c[99] + slope * (t - 100)))
        else:
            out[arm] = (lambda t, sk=c[99] / 100.0: sk * max(t, 0))
    return s_pol, out


def wall_budget(curve: Any, T: float) -> int:
    t = 0
    while curve(t + 1) <= T:
        t += 1
    return t


def _improvements(trace: dict[str, Any], limit: int) -> list[list[float]]:
    out, best = [], np.inf
    for k in np.argsort(trace["units"], kind="stable"):
        u, v = int(trace["units"][k]), float(trace["values"][k])
        if u <= limit and v < best:
            best = v
            out.append([u, v])
    return out


def wall_auc(imp: list[list[float]], curve: Any, T: float, b_k: int, f_star: float, tau: float,
             scale: float) -> float:
    """(1/T) ∫_0^T L dτ, L piecewise constant between evaluation completion times."""
    fbest = np.full(b_k + 1, np.inf)  # index t: best after t units
    for u, v in imp:
        if u <= b_k:
            fbest[int(u)] = min(fbest[int(u)], v)
    fbest = np.minimum.accumulate(fbest)
    levels = bl.clipped_log_regret(fbest, f_star, tau, scale)
    total = 0.0
    for t in range(b_k + 1):
        start = min(curve(t), T)
        end = min(curve(t + 1), T) if t < b_k else T
        total += levels[t] * max(end - start, 0.0)
    return total / T


INSTANCES = 200


def tune(name: str, progress: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    s_pol, arm_curves = curves(name)
    fam = b0.Family(name, INSTANCES)
    ref = next(f for f in json.loads(B0_RESULTS.read_text())["families"] if f["family"] == name)
    # f*, scale and tolerance as in B-S2 (f* from its reference; synthetic f* = 0)
    run_dir = ROOT / "runs"
    f_star = scale = None
    for d in sorted(run_dir.glob("v08b-b0-*/" + name + "/units/reference.json")):
        r = json.loads(d.read_text())["row"]
        f_star = np.asarray(r["f_star"])[:INSTANCES]
        scale = np.asarray(r["scale"])[:INSTANCES]
    if f_star is None:
        raise SystemExit(f"B-S2 reference for {name} not found under runs/")
    tau = b0.EPS_ABS + b0.EPS_REL * np.abs(f_star)
    budgets = {arm: {B: wall_budget(c, s_pol * B) for B in b0.BUDGETS}
               for arm, c in arm_curves.items()}
    polisher = b0.Polisher(fam)
    results: dict[str, Any] = {}
    for j, arm in enumerate(arm_curves):
        progress(f"arm {arm}", j, len(arm_curves))
        long = max(budgets[arm].values())
        configs = _configs(arm, fam)
        results[arm] = {}
        for cfg, runner in configs.items():
            per_budget = {str(B): {"auc_wall": [], "basin": []} for B in b0.BUDGETS}
            for seed in (b0.SEEDS if arm in ("quasi-random", "cmaes", "tpe") else (0,)):
                for B in b0.BUDGETS:
                    b_k = budgets[arm][B]
                    T = s_pol * B
                    if b_k < 1:
                        per_budget[str(B)]["auc_wall"].append([1.0] * fam.n)
                        per_budget[str(B)]["basin"].append([False] * fam.n)
                        continue
                    traces = runner(b_k if arm in ("owned-ms",) else long, seed, b_k)
                    aucs = [wall_auc(_improvements(tr, b_k), arm_curves[arm], T, b_k,
                                     f_star[i], tau[i], scale[i]) for i, tr in enumerate(traces)]
                    best = [(i, bl.best_point(tr, b_k)) for i, tr in enumerate(traces)]
                    pol = polisher([(i, x if x is not None else fam.x0[i]) for i, x in best])
                    per_budget[str(B)]["auc_wall"].append(aucs)
                    per_budget[str(B)]["basin"].append((np.asarray(pol) <= f_star + tau).tolist())
            results[arm][cfg] = {
                B: {"auc_wall_mean": float(np.mean(np.mean(v["auc_wall"], axis=0))),
                    "basin_rate": float(np.mean(np.mean(v["basin"], axis=0))),
                    "auc_wall_per_instance": np.mean(v["auc_wall"], axis=0).tolist(),
                    "basin_per_instance": np.mean(v["basin"], axis=0).tolist()}
                for B, v in per_budget.items()}
    tuned = {str(B): {arm: {
        "by_auc": min(cfgs, key=lambda c, B=B: cfgs[c][str(B)]["auc_wall_mean"]),
        "by_basin": max(cfgs, key=lambda c, B=B: cfgs[c][str(B)]["basin_rate"]),
        "wall_budget_units": budgets[arm][B]} for arm, cfgs in results.items()}
        for B in b0.BUDGETS}
    for B in b0.BUDGETS:
        for arm, t in tuned[str(B)].items():
            t["auc_wall_mean"] = results[arm][t["by_auc"]][str(B)]["auc_wall_mean"]
            t["basin_rate"] = results[arm][t["by_basin"]][str(B)]["basin_rate"]
    return {"family": name, "policy_seconds_per_unit": s_pol, "wall_budgets": budgets,
            "tuned": tuned, "results": results, "b0_reference_method": ref["reference"]["method"],
            "wall_seconds": time.perf_counter() - t0}


def _configs(arm: str, fam: Any) -> dict[str, Any]:
    """Runner(run_budget, seed, b_k) -> traces for each configuration of the B-S2 grid."""
    n = fam.n
    if arm == "quasi-random":
        return {"sobol": lambda L, s, bk: [bl.quasi_random(fam.f_batch(i), fam.x0[i], fam.lo,
                                                           fam.hi, L, seed=s) for i in range(n)]}
    if arm == "cmaes":
        pop0 = 4 + int(3 * np.log(fam.d))
        return {f"sigma0={sg},popsize={'default' if p is None else '2x'}":
                (lambda L, s, bk, sg=sg, p=p: [bl.cmaes(fam.f_batch(i), fam.x0[i], fam.lo, fam.hi,
                                                        L, sg, None if p is None else 2 * pop0, s)
                                               for i in range(n)])
                for sg, p in b0.CMA_GRID}
    if arm == "tpe":
        return {f"n_startup={st},multivariate={mv}":
                (lambda L, s, bk, st=st, mv=mv: [bl.tpe(fam.f(i), fam.x0[i], fam.lo, fam.hi, L,
                                                        st, mv, s) for i in range(n)])
                for st, mv in b0.TPE_GRID}
    if arm == "bb":
        return {f"fallback={g}": (lambda L, s, bk, g=g: bl.bb(
                    fam.objective, fam.contexts, fam.x0, fam.lo, fam.hi, max(L, 2), g))
                for g in b0.BB_GRID}

    def multistart(k: int, kind: str) -> Any:
        def runner(L: int, s: int, bk: int) -> list[Any]:
            per = max(bk // k, 2) if kind == "owned" else bk // k
            starts = [fam.start(j) for j in range(k)]
            if kind == "owned":
                per_start = [bl.owned_lbfgs(fam.objective, fam.contexts, x, fam.lo, fam.hi, per)
                             for x in starts]
            else:
                per_start = [[bl.scipy_lbfgsb(fam.vg(i), x[i], fam.lo, fam.hi, max(per, 2))
                              for i in range(n)] for x in starts]
            return [bl.combine_multistart([per_start[j][i] for j in range(k)], per)
                    for i in range(n)]
        return runner

    kind = "owned" if arm == "owned-ms" else "scipy"
    return {f"K={k}": multistart(k, kind) for k in b0.K_GRID}


def run(name: str, run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    code = run_checkpoint.fingerprint(RUN_FILES)
    with (run_dir / "command.txt").open("a") as handle:
        handle.write(run_checkpoint.command_record(
            [sys.executable, *sys.argv],
            {"code_fingerprint": code, "cap": "1.5 CPU-hours (process CPU clock; one pinned "
                                              "core); 4 CPU-hours for the three families"}))
    run_checkpoint.run_units([name], run_dir, tune, fingerprint=code)
    (run_dir / "DONE").write_text("units/\n")
    return 0


def combine(dirs: list[Path], output: Path) -> int:
    if output.exists():
        raise SystemExit(f"{output} exists; results are never overwritten")
    rows = [json.loads(p.read_text())["row"]
            for d in dirs for p in sorted((d / "units").glob("*.json"))]
    run_checkpoint.write_atomic(output, json.dumps({"schema": "neuralopt-v08b-wall-tuning/v1",
                                                    "families": rows}) + "\n")
    print(f"wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=("rastrigin", "ackley", "sinusoid"), required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("combine")
    c.add_argument("--runs", type=Path, nargs="+", required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.family, args.out) if args.command == "run" else combine(args.runs, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
