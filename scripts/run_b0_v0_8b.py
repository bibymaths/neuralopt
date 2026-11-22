"""v0.8-B step B-S2 (no learning): baselines, anytime regret and global-basin rates on the
validation split of each v0.8-B family. Harness only.

Plan: docs/plans/neuralopt-v08b-step-policy.md (sections 2-6, 11).

Subcommands:
  run        one family into a run directory (checkpoint per arm; rerun resumes)
  summarize  combine finished family run directories into one versioned results file

Arms (validation only; budgets read off one run per configuration and seed):
  quasi-random   scrambled Sobol, first point x0; 5 seeds
  owned-ms       owned L-BFGS multistart, K in {1, 2, 4, 8}, B/K units per start
  scipy-ms       SciPy L-BFGS-B multistart (full-budget tolerances), K in {1, 2, 4, 8}
  bb             NeuralOpt's compiled BB loop; fallback (initial) length in {1e-3, 1e-2, 1e-1}
  cmaes          pycma; sigma0 in {0.1, 0.2, 0.3} of the u-box width, popsize {default, 2x}; 5 seeds
  tpe            optuna TPE; n_startup in {5, 10}, multivariate in {False, True}; 5 seeds
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
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_checkpoint  # noqa: E402
import v08b_baselines as bl  # noqa: E402

from neuralopt.families import FAMILIES, SYNTHETIC, family_objective, make_split  # noqa: E402

FAMILY_NAMES = ("rastrigin", "ackley", "rosenbrock-shifted", "sinusoid", "cascade-realistic")
BUDGETS = (20, 50, 100)
LONG = max(BUDGETS)
SEEDS = (0, 1, 2, 3, 4)
K_GRID = (1, 2, 4, 8)
BB_GRID = (1e-3, 1e-2, 1e-1)
CMA_GRID = tuple((s, p) for s in (0.1, 0.2, 0.3) for p in (None, "2x"))
TPE_GRID = tuple((n, m) for n in (5, 10) for m in (False, True))
EPS_ABS, EPS_REL = 1e-6, 1e-3
NORMALIZATION_POINTS, NORMALIZATION_SEED = 64, 101
ARM_START_SEED = 3
HEADROOM_RATE = 0.95
ARMS = ("quasi-random", "owned-ms", "scipy-ms", "bb", "cmaes", "tpe")
RUN_FILES = (ROOT / "scripts" / "run_b0_v0_8b.py", ROOT / "scripts" / "v08b_baselines.py",
             ROOT / "scripts" / "run_checkpoint.py", ROOT / "scripts" / "run_f0_v0_8.py",
             ROOT / "src" / "neuralopt" / "families.py", ROOT / "src" / "neuralopt" / "lbfgs.py",
             ROOT / "src" / "neuralopt" / "compiled.py", ROOT / "uv.lock")
POLISH_CHUNK = 200


# ---- family setup ---------------------------------------------------------------------


class Family:
    def __init__(self, name: str, n: int) -> None:
        self.name, self.n = name, n
        self.contexts, self.truths = make_split(name, "validation", n)
        self.objective = family_objective(name)
        if name in SYNTHETIC:
            lo, hi = SYNTHETIC[name].bounds("validation")
            self.x0 = np.asarray(self.contexts["x0"])
        else:
            spec = FAMILIES[name]
            lo, hi = spec.lo, spec.hi
            self.x0 = np.tile(spec.default, (n, 1))
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.d = self.lo.size
        self.ctx = [jax.tree_util.tree_map(lambda a, i=i: a[i], self.contexts) for i in range(n)]
        self._fb = jax.jit(jax.vmap(self.objective, in_axes=(0, None)))
        self._f = jax.jit(self.objective)
        self._vg = jax.jit(jax.value_and_grad(self.objective))
        self.starts = self._sobol(max(K_GRID) - 1, ARM_START_SEED)

    def _sobol(self, n: int, seed: int) -> np.ndarray:
        from scipy.stats import qmc

        m = int(np.ceil(np.log2(max(n, 2))))
        return self.lo + (self.hi - self.lo) * qmc.Sobol(self.d, scramble=True,
                                                         seed=seed).random_base2(m)[:n]

    def f_batch(self, i: int) -> Any:
        return lambda xs: np.asarray(self._fb(jnp.asarray(np.atleast_2d(xs)), self.ctx[i]))

    def f(self, i: int) -> Any:
        return lambda x: float(self._f(jnp.asarray(x), self.ctx[i]))

    def vg(self, i: int) -> Any:
        def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            v, g = self._vg(jnp.asarray(x), self.ctx[i])
            return float(v), np.asarray(g)
        return value_and_grad

    def start(self, k: int) -> np.ndarray:
        """Start k for every instance: k = 0 is x0_i, k >= 1 the shared Sobol starts."""
        return self.x0 if k == 0 else np.tile(self.starts[k - 1], (self.n, 1))


# ---- per-arm results ------------------------------------------------------------------


def _improvements(trace: dict[str, Any]) -> list[list[float]]:
    """Best-so-far change points within LONG units: [[unit, value], ...] (compact)."""
    out, best = [], np.inf
    order = np.argsort(trace["units"], kind="stable")
    for k in order:
        u, v = int(trace["units"][k]), float(trace["values"][k])
        if u <= LONG and v < best:
            best = v
            out.append([u, v])
    return out


class Polisher:
    """Uncharged polish of best points, deduplicated, in fixed-size chunks (one compile)."""

    def __init__(self, fam: Family) -> None:
        self.fam, self.cache = fam, {}
        self.chunk = min(POLISH_CHUNK, fam.n)

    def __call__(self, points: list[tuple[int, np.ndarray]]) -> list[float]:
        keys = [(i, np.asarray(x, float).tobytes()) for i, x in points]
        todo = [(k, i, x) for k, (i, x) in zip(keys, points, strict=True) if k not in self.cache]
        unique = list({k: (k, i, x) for k, i, x in todo}.values())
        for start in range(0, len(unique), self.chunk):
            chunk = unique[start:start + self.chunk]
            pad = chunk + [chunk[-1]] * (self.chunk - len(chunk))
            idx = np.array([i for _, i, _ in pad])
            ctx = jax.tree_util.tree_map(lambda a, idx=idx: a[idx], self.fam.contexts)
            xs = np.stack([x for _, _, x in pad])
            values = bl.polish(self.fam.objective, ctx, xs, self.fam.lo, self.fam.hi)
            for (k, _, _), v in zip(chunk, values[:len(chunk)], strict=True):
                self.cache[k] = float(v)
        return [self.cache[k] for k in keys]


def _record(fam: Family, polisher: Polisher, traces: dict[int, dict[str, Any]],
            calls: dict[int, list[tuple[int, int]]]) -> dict[str, Any]:
    """Per budget-keyed traces (``traces[B][i]``, built for budget B) -> stored record."""
    out: dict[str, Any] = {"improvements": {}, "polished": {}, "calls": {}}
    for b, per_instance in traces.items():
        out["improvements"][str(b)] = [_improvements(t) for t in per_instance]
        best = [(i, bl.best_point(t, b)) for i, t in enumerate(per_instance)]
        out["polished"][str(b)] = polisher([(i, x if x is not None else fam.x0[i])
                                            for i, x in best])
        out["calls"][str(b)] = [list(c) for c in calls[b]]
    return out


def arm_quasi_random(fam: Family, polisher: Polisher, progress: Any) -> dict[str, Any]:
    runs = {}
    for s in SEEDS:
        progress(f"quasi-random seed {s}", s, len(SEEDS))
        t0 = time.perf_counter()
        tr = [bl.quasi_random(fam.f_batch(i), fam.x0[i], fam.lo, fam.hi, LONG, seed=s)
              for i in range(fam.n)]
        wall = time.perf_counter() - t0
        rec = _record(fam, polisher, {b: tr for b in BUDGETS},
                      {b: [bl.calls_within(t, b) for t in tr] for b in BUDGETS})
        rec["wall_seconds"] = wall
        runs[f"seed={s}"] = rec
    return {"configs": {"sobol": runs}}


def _multistart(fam: Family, polisher: Polisher, progress: Any, kind: str) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
    walls: dict[tuple[int, int], float] = {}

    def start_traces(k: int, b: int) -> list[dict[str, Any]]:
        key = (k, b) if kind == "owned" else (k, LONG)
        if key not in cache:
            t0 = time.perf_counter()
            if kind == "owned":
                cache[key] = bl.owned_lbfgs(fam.objective, fam.contexts, fam.start(k), fam.lo,
                                            fam.hi, max(b, 2))
            else:
                x0 = fam.start(k)
                cache[key] = [bl.scipy_lbfgsb(fam.vg(i), x0[i], fam.lo, fam.hi, LONG)
                              for i in range(fam.n)]
            walls[key] = time.perf_counter() - t0
        return cache[key]

    for j, kk in enumerate(K_GRID):
        progress(f"{kind} multistart K={kk}", j, len(K_GRID))
        traces, calls = {}, {}
        for b in BUDGETS:
            per = b // kk
            per_start = [start_traces(k, per) for k in range(kk)]
            combined = [bl.combine_multistart([per_start[k][i] for k in range(kk)], per)
                        for i in range(fam.n)]
            traces[b] = combined
            if kind == "owned":
                calls[b] = [(sum(per_start[k][i]["objective_calls"] for k in range(kk)),
                             sum(per_start[k][i]["gradient_calls"] for k in range(kk)))
                            for i in range(fam.n)]
            else:
                calls[b] = [bl.calls_within(t, b) for t in combined]
        rec = _record(fam, polisher, traces, calls)
        configs[f"K={kk}"] = {"seed=0": rec}
    return {"configs": configs, "wall_seconds_by_start_and_budget":
            {f"{k}/{b}": w for (k, b), w in walls.items()}}


def arm_owned(fam: Family, polisher: Polisher, progress: Any) -> dict[str, Any]:
    return _multistart(fam, polisher, progress, "owned")


def arm_scipy(fam: Family, polisher: Polisher, progress: Any) -> dict[str, Any]:
    return _multistart(fam, polisher, progress, "scipy")


def arm_bb(fam: Family, polisher: Polisher, progress: Any) -> dict[str, Any]:
    configs = {}
    for j, length in enumerate(BB_GRID):
        progress(f"bb fallback={length}", j, len(BB_GRID))
        t0 = time.perf_counter()
        tr = bl.bb(fam.objective, fam.contexts, fam.x0, fam.lo, fam.hi, LONG, length)
        wall = time.perf_counter() - t0
        rec = _record(fam, polisher, {b: tr for b in BUDGETS},
                      {b: [bl.calls_within(t, b) for t in tr] for b in BUDGETS})
        rec["wall_seconds"] = wall
        configs[f"fallback={length}"] = {"seed=0": rec}
    return {"configs": configs}


def arm_cmaes(fam: Family, polisher: Polisher, progress: Any) -> dict[str, Any]:
    default_pop = 4 + int(3 * np.log(fam.d))
    configs = {}
    for j, (sigma, pop) in enumerate(CMA_GRID):
        runs = {}
        popsize = None if pop is None else 2 * default_pop
        for s in SEEDS:
            progress(f"cmaes sigma0={sigma} pop={pop} seed {s}", j * len(SEEDS) + s,
                     len(CMA_GRID) * len(SEEDS))
            t0 = time.perf_counter()
            tr = [bl.cmaes(fam.f_batch(i), fam.x0[i], fam.lo, fam.hi, LONG, sigma, popsize, s)
                  for i in range(fam.n)]
            wall = time.perf_counter() - t0
            rec = _record(fam, polisher, {b: tr for b in BUDGETS},
                          {b: [bl.calls_within(t, b) for t in tr] for b in BUDGETS})
            rec["wall_seconds"] = wall
            runs[f"seed={s}"] = rec
        configs[f"sigma0={sigma},popsize={'default' if pop is None else '2x'}"] = runs
    return {"configs": configs, "default_popsize": default_pop}


def arm_tpe(fam: Family, polisher: Polisher, progress: Any) -> dict[str, Any]:
    configs = {}
    for j, (startup, multivariate) in enumerate(TPE_GRID):
        runs = {}
        for s in SEEDS:
            progress(f"tpe startup={startup} mv={multivariate} seed {s}", j * len(SEEDS) + s,
                     len(TPE_GRID) * len(SEEDS))
            t0 = time.perf_counter()
            tr = [bl.tpe(fam.f(i), fam.x0[i], fam.lo, fam.hi, LONG, startup, multivariate, s)
                  for i in range(fam.n)]
            wall = time.perf_counter() - t0
            rec = _record(fam, polisher, {b: tr for b in BUDGETS},
                          {b: [bl.calls_within(t, b) for t in tr] for b in BUDGETS})
            rec["wall_seconds"] = wall
            runs[f"seed={s}"] = rec
        configs[f"n_startup={startup},multivariate={multivariate}"] = runs
    return {"configs": configs}


ARM_FUNCTIONS = {"quasi-random": arm_quasi_random, "owned-ms": arm_owned, "scipy-ms": arm_scipy,
                 "bb": arm_bb, "cmaes": arm_cmaes, "tpe": arm_tpe}


def reference(fam: Family, progress: Any) -> dict[str, Any]:
    """f* per instance, the normalization scale s_i and the zero-search polish."""
    t0 = time.perf_counter()
    if fam.name in SYNTHETIC:
        f_star = np.zeros(fam.n)
        method, reproduction = "known: f* = 0 at x = shift", 1.0
    else:
        import run_f0_v0_8 as f0

        spec = FAMILIES[fam.name]
        runner = f0._ScipyRunner(fam.name)
        best = {}
        for tag in ("reference_a", "reference_b"):
            starts = f0._sobol(spec, f0.REFERENCE_STARTS, f0.SOBOL_SEEDS[tag])
            vals = np.full(fam.n, np.inf)
            for i in range(fam.n):
                progress(f"reference {tag}", i, fam.n)
                finals = [tr[-1][1] for x0 in [*starts, np.asarray(fam.truths[i])]
                          if (tr := runner.trace(x0, fam.ctx[i], f0.REFERENCE_UNITS,
                                                 precise=True))]
                vals[i] = min(finals)
            best[tag] = vals
        f_star = np.minimum(best["reference_a"], best["reference_b"])
        tol = EPS_ABS + EPS_REL * np.abs(f_star)
        reproduction = float(np.mean(np.abs(best["reference_a"] - best["reference_b"]) <= tol))
        method = ("F0 protocol: SciPy L-BFGS-B, 64 Sobol starts + truth, 2 seeds, "
                  "cap 2000 units, ftol 1e-15, gtol 1e-10")
    pts = fam._sobol(NORMALIZATION_POINTS, NORMALIZATION_SEED)
    scale = np.array([float(np.median(fam.f_batch(i)(pts) - f_star[i])) for i in range(fam.n)])
    zero_search = Polisher(fam)([(i, fam.x0[i]) for i in range(fam.n)])
    return {"method": method, "two_seed_reproduction_rate": reproduction,
            "f_star": f_star.tolist(), "scale": scale.tolist(),
            "x0_polished": zero_search, "wall_seconds": time.perf_counter() - t0}


# ---- summary --------------------------------------------------------------------------


def _auc_from_improvements(imp: list[list[float]], budget: int, f_star: float, tau: float,
                           scale: float) -> float:
    fbest = np.full(budget, np.inf)
    for u, v in imp:
        if u <= budget:
            fbest[int(u) - 1] = min(fbest[int(u) - 1], v)
    return float(np.mean(bl.clipped_log_regret(np.minimum.accumulate(fbest), f_star, tau, scale)))


def summarize_family(fam_name: str, ref: dict[str, Any], arms: dict[str, Any]) -> dict[str, Any]:
    f_ref = np.asarray(ref["f_star"])
    tau_ref = EPS_ABS + EPS_REL * np.abs(f_ref)
    lowest = f_ref.copy()
    for arm in arms.values():
        for runs in arm["configs"].values():
            for rec in runs.values():
                for b in BUDGETS:
                    for i, imp in enumerate(rec["improvements"][str(b)]):
                        if imp:
                            lowest[i] = min(lowest[i], imp[-1][1])
                    lowest = np.minimum(lowest, np.asarray(rec["polished"][str(b)]))
    updated = lowest < f_ref - tau_ref
    f_star = np.minimum(f_ref, lowest)
    tau = EPS_ABS + EPS_REL * np.abs(f_star)
    scale = np.asarray(ref["scale"])
    valid = scale > tau
    table: dict[str, Any] = {}
    per_instance: dict[str, Any] = {}
    for arm_name, arm in arms.items():
        table[arm_name] = {}
        for cfg, runs in arm["configs"].items():
            row: dict[str, Any] = {}
            for b in BUDGETS:
                auc = np.array([[_auc_from_improvements(imp, b, f_star[i], tau[i], scale[i])
                                 for i, imp in enumerate(rec["improvements"][str(b)])]
                                for rec in runs.values()]).mean(axis=0)
                basin = np.array([np.asarray(rec["polished"][str(b)]) <= f_star + tau
                                  for rec in runs.values()]).mean(axis=0)
                calls = np.array([rec["calls"][str(b)] for rec in runs.values()], float)
                row[str(b)] = {"auc_mean": float(auc[valid].mean()),
                               "basin_rate": float(basin[valid].mean()),
                               "objective_calls_mean": float(calls[..., 0].mean()),
                               "gradient_calls_mean": float(calls[..., 1].mean())}
                per_instance[f"{arm_name}|{cfg}|{b}"] = {"auc": auc.tolist(),
                                                         "basin": basin.tolist()}
            walls = [rec.get("wall_seconds") for rec in runs.values()]
            if all(w is not None for w in walls):
                row["wall_seconds_per_instance_mean"] = float(np.mean(walls)) / len(f_star)
            table[arm_name][cfg] = row
    tuned: dict[str, Any] = {}
    for b in BUDGETS:
        tuned[str(b)] = {}
        for arm_name, cfgs in table.items():
            by_auc = min(cfgs, key=lambda c, b=b: cfgs[c][str(b)]["auc_mean"])
            by_basin = max(cfgs, key=lambda c, b=b: cfgs[c][str(b)]["basin_rate"])
            tuned[str(b)][arm_name] = {
                "by_auc": by_auc, "auc_mean": cfgs[by_auc][str(b)]["auc_mean"],
                "by_basin": by_basin, "basin_rate": cfgs[by_basin][str(b)]["basin_rate"]}
    best20 = max(v["basin_rate"] for v in tuned["20"].values())
    spreads = {}
    for b in BUDGETS:
        ranked = sorted(tuned[str(b)].items(), key=lambda kv: kv[1]["auc_mean"])
        (a1, t1), (a2, t2) = ranked[0], ranked[1]
        auc1 = np.asarray(per_instance[f"{a1}|{t1['by_auc']}|{b}"]["auc"])[valid]
        auc2 = np.asarray(per_instance[f"{a2}|{t2['by_auc']}|{b}"]["auc"])[valid]
        rb = sorted(tuned[str(b)].items(), key=lambda kv: -kv[1]["basin_rate"])
        (c1, u1), (c2, u2) = rb[0], rb[1]
        bas1 = np.asarray(per_instance[f"{c1}|{u1['by_basin']}|{b}"]["basin"])[valid]
        bas2 = np.asarray(per_instance[f"{c2}|{u2['by_basin']}|{b}"]["basin"])[valid]
        spreads[str(b)] = {
            "auc_best_two": [a1, a2], "sd_auc_difference_best_two": float(np.std(auc1 - auc2,
                                                                                 ddof=1)),
            "sd_auc_best": float(np.std(auc1, ddof=1)),
            "basin_best_two": [c1, c2],
            "sd_basin_difference_best_two": float(np.std(bas1 - bas2, ddof=1)),
            "sd_basin_best": float(np.std(bas1, ddof=1)),
        }
    return {
        "family": fam_name,
        "instances": len(f_star), "instances_excluded_scale_le_tau": int((~valid).sum()),
        "reference": {k: ref[k] for k in ("method", "two_seed_reproduction_rate",
                                          "wall_seconds")},
        "f_star_updated_by_arms": int(updated.sum()),
        "zero_search_basin_rate": float(np.mean(np.asarray(ref["x0_polished"]) <= f_star + tau)),
        "table": table, "tuned": tuned, "spreads_for_power": spreads,
        "headroom": {"rule": f"not tested if the best baseline's basin rate at B=20 >= "
                             f"{HEADROOM_RATE}", "best_basin_rate_at_20": best20,
                     "no_headroom": bool(best20 >= HEADROOM_RATE)},
    }


# ---- commands -------------------------------------------------------------------------


def run(name: str, run_dir: Path, n: int) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "family-result.json"
    if final.exists():
        raise SystemExit(f"{final} exists; results are never overwritten")
    code = run_checkpoint.fingerprint(RUN_FILES)
    with (run_dir / "command.txt").open("a") as handle:
        handle.write(run_checkpoint.command_record(
            [sys.executable, *sys.argv],
            {"code_fingerprint": code, "instances": str(n),
             "cap": "3 CPU-hours for this process (process CPU clock; one pinned core, so "
                    "CPU <= wall); B-S2 total cap 10 CPU-hours"}))
    fam = Family(name, n)
    print(json.dumps({"family": name, "instances": n, "dimension": fam.d}), flush=True)

    def compute(unit: str, progress: Any) -> dict[str, Any]:
        t0, c0 = time.perf_counter(), time.process_time()
        result = (reference(fam, progress) if unit == "reference"
                  else ARM_FUNCTIONS[unit](fam, Polisher(fam), progress))
        result["unit_wall_seconds"] = time.perf_counter() - t0
        result["unit_cpu_seconds"] = time.process_time() - c0
        print(json.dumps({"unit": unit, "wall_s": round(result["unit_wall_seconds"], 1),
                          "cpu_s": round(result["unit_cpu_seconds"], 1)}), flush=True)
        return result

    rows = run_checkpoint.run_units(["reference", *ARMS], run_dir, compute, fingerprint=code,
                                    weights={"reference": 3.0, "tpe": 3.0},
                                    min_interval=30.0)
    ref, arms = rows[0], dict(zip(ARMS, rows[1:], strict=True))
    summary = summarize_family(name, ref, arms)
    summary["cpu_seconds_total"] = sum(r["unit_cpu_seconds"] for r in rows)
    summary["wall_seconds_total"] = sum(r["unit_wall_seconds"] for r in rows)
    summary["code_fingerprint"] = code
    run_checkpoint.write_atomic(final, json.dumps(summary, indent=1) + "\n")
    (run_dir / "DONE").write_text(final.name + "\n")
    print(json.dumps({"family": name, "headroom": summary["headroom"],
                      "cpu_s": round(summary["cpu_seconds_total"])}), flush=True)
    return 0


def summarize(dirs: list[Path], output: Path) -> int:
    if output.exists():
        raise SystemExit(f"{output} exists; results are never overwritten")
    families = []
    for d in dirs:
        if not (d / "DONE").exists():
            raise SystemExit(f"{d} has not finished")
        families.append(json.loads((d / "family-result.json").read_text()))
    report = {"schema": "neuralopt-v08b-b0/v1", "budgets": BUDGETS, "seeds": SEEDS,
              "eps_abs": EPS_ABS, "eps_rel": EPS_REL, "headroom_rate": HEADROOM_RATE,
              "environment": {"jax": jax.__version__, "xla_flags": os.environ.get("XLA_FLAGS")},
              "cpu_hours_total": sum(f["cpu_seconds_total"] for f in families) / 3600,
              "families": families}
    run_checkpoint.write_atomic(output, json.dumps(report, indent=1) + "\n")
    print(f"wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=FAMILY_NAMES, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--instances", type=int, default=200)
    s = sub.add_parser("summarize")
    s.add_argument("--runs", type=Path, nargs="+", required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        return run(args.family, args.out, args.instances)
    return summarize(args.runs, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
