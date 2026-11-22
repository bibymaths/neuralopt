"""v0.8 gate F0 (no learning): headroom per family. Harness only.

Subcommands:
  probe   timing probe: warm µs per objective/gradient call (owned batched, SciPy per call),
          units per local solve; projects F0 compute against the 6 CPU-hour cap.
  run     the F0 study (written after the probe is reviewed). With ``--out RUN_DIR`` each
          finished family is checkpointed in RUN_DIR and a rerun with the same RUN_DIR resumes
          (scripts/run_checkpoint.py); progress.json is rewritten atomically.

See docs/plans/neuralopt-v08-amortized-families.md, section 9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuralopt.families import FAMILIES, family_objective, make_split  # noqa: E402
from neuralopt.lbfgs import lbfgs_solve_batch  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_checkpoint  # noqa: E402

VARIANTS = ("sinusoid", "cascade-clean", "cascade-realistic")
N_VALIDATION, N_TRAIN = 200, 2000
REFERENCE_STARTS, REFERENCE_UNITS = 64, 2000
LABEL_STARTS, LABEL_UNITS = 16, 1000
BUDGETS = (25, 50, 100, 200, 400, 800)
CAP_CPU_HOURS = 6.0


def _timeit(fn: Any, repeats: int = 5) -> float:
    jax.block_until_ready(fn())  # compile + warm
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def probe_family(name: str) -> dict[str, Any]:
    spec = FAMILIES[name]
    f = family_objective(name)
    contexts, truths = make_split(name, "validation", 64)
    n = truths.shape[0]
    rng = np.random.default_rng(0)
    x = jnp.asarray(spec.lo + (spec.hi - spec.lo) * rng.uniform(size=(n, spec.dimension)))
    obj = jax.jit(jax.vmap(f))
    grad = jax.jit(jax.vmap(jax.grad(f)))
    us_obj = 1e6 * _timeit(lambda: obj(x, contexts)) / n
    us_grad = 1e6 * _timeit(lambda: grad(x, contexts)) / n

    # SciPy-style single calls: jitted value_and_grad, one instance, host round trip
    vg = jax.jit(jax.value_and_grad(f))
    c0 = jax.tree_util.tree_map(lambda a: a[0], contexts)
    vg(x[0], c0)
    t0 = time.perf_counter()
    for i in range(200):
        v, g = vg(np.asarray(x[i % n]), c0)
        float(v), np.asarray(g)
    us_scipy_call = 1e6 * (time.perf_counter() - t0) / 200

    # owned L-BFGS from random starts, cap 2000 units: units per local solve and wall time
    lo, hi = jnp.tile(spec.lo, (n, 1)), jnp.tile(spec.hi, (n, 1))
    per_unit, first_call = {}, {}
    for strategy in ("vmap", "map"):
        t0 = time.perf_counter()
        out = lbfgs_solve_batch(f, contexts, x, (lo, hi), REFERENCE_UNITS, strategy=strategy)
        first_call[strategy] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = lbfgs_solve_batch(f, contexts, x, (lo, hi), REFERENCE_UNITS, strategy=strategy)
        per_unit[strategy] = 1e6 * (time.perf_counter() - t0) / np.asarray(out["used"]).sum()
    used = np.asarray(out["used"])
    best = min(per_unit, key=per_unit.__getitem__)
    return {
        "family": name,
        "us_per_objective_batched": us_obj,
        "us_per_gradient_batched": us_grad,
        "gradient_to_objective_cost": us_grad / us_obj,
        "us_per_scipy_call_value_and_grad": us_scipy_call,
        "local_solve_units_median": float(np.median(used)),
        "local_solve_units_p90": float(np.percentile(used, 90)),
        "local_solve_hit_cap_fraction": float(np.mean(used >= REFERENCE_UNITS - 1)),
        "owned_us_per_unit": per_unit,
        "owned_first_call_s": first_call,
        "owned_strategy": best,
        "owned_batched_us_per_unit": per_unit[best],
    }


def project(p: dict[str, Any]) -> dict[str, Any]:
    """CPU-seconds for F0, using the measured units per local solve."""
    per_start = p["local_solve_units_median"]
    owned = p["owned_batched_us_per_unit"] * 1e-6
    scipy_call = p["us_per_scipy_call_value_and_grad"] * 1e-6  # one call = 2 units
    # SciPy reference: 2 seeds x (64 + truth) precise starts; assume 2x the units of a
    # default-tolerance local solve for the precise (ftol 1e-15, gtol 1e-10) runs.
    refs = 2 * N_VALIDATION * (REFERENCE_STARTS + 1) * (2 * per_start / 2) * scipy_call
    labels = 2 * N_TRAIN * LABEL_STARTS * min(per_start, LABEL_UNITS) * owned  # + adequacy
    owned_arms = (2 + 16) * N_VALIDATION * per_start * owned  # A0, A3, A2o (16 starts)
    scipy_arms = (1 + 16) * N_VALIDATION * per_start / 2 * scipy_call  # A1, A2 (16 starts)
    identifiability = 50 * 17 * (2 * per_start / 2) * scipy_call
    total = refs + labels + owned_arms + scipy_arms + identifiability
    return {"scipy_references_s": refs, "training_labels_s": labels,
            "owned_arms_s": owned_arms, "scipy_arms_s": scipy_arms,
            "identifiability_s": identifiability, "total_cpu_hours": total / 3600}


# --- F0 run --------------------------------------------------------------------------

EPS_ABS, EPS_REL = 1e-6, 1e-3  # success tolerance, fixed before F0 runs
FISHER_COND_LIMIT = 1e6  # cond(J^T J) above this marks a non-identified direction
MULTISTART_K = (2, 4, 8, 16)  # multistart start counts tried at each budget (validation)
LONG = max(BUDGETS)  # one run per start at the largest budget; smaller budgets read off it
SOBOL_SEEDS = {"reference_a": 1, "reference_b": 2, "arms": 3}


def _sobol(spec: Any, n: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    u = qmc.Sobol(spec.dimension, scramble=True, seed=seed).random_base2(int(np.log2(n)))
    return spec.lo + (spec.hi - spec.lo) * u


def _success(value: float, f_star: float) -> bool:
    return value - f_star <= EPS_ABS + EPS_REL * abs(f_star)


class _ScipyRunner:
    """SciPy L-BFGS-B per start; one call = objective + gradient = 2 units (harness)."""

    def __init__(self, name: str) -> None:
        self.spec = FAMILIES[name]
        self.vg = jax.jit(jax.value_and_grad(family_objective(name)))
        self.calls = 0

    def trace(self, x0: np.ndarray, c: Any, max_units: int, precise: bool = False
              ) -> list[tuple[int, float]]:
        from scipy.optimize import minimize as scipy_minimize

        trace: list[tuple[int, float]] = []

        class Exhausted(Exception):
            pass

        def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
            if 2 * (len(trace) + 1) > max_units:
                raise Exhausted
            v, g = self.vg(jnp.asarray(x), c)
            v = float(v)
            v = v if np.isfinite(v) else np.inf
            best = min(v, trace[-1][1] if trace else np.inf)
            trace.append((2 * (len(trace) + 1), best))
            self.calls += 1
            g = np.asarray(g, float)
            return v, np.where(np.isfinite(g), g, 0.0)

        options = {"maxfun": max_units // 2, "maxiter": max_units}
        if precise:
            options.update(ftol=1e-15, gtol=1e-10)
        try:
            scipy_minimize(fun, np.clip(x0, self.spec.lo, self.spec.hi), jac=True,
                           method="L-BFGS-B", bounds=list(zip(self.spec.lo, self.spec.hi,
                                                              strict=True)), options=options)
        except Exhausted:
            pass
        return trace


def _best_within(trace: list[tuple[int, float]], units: int) -> float:
    best = np.inf
    for u, v in trace:
        if u > units:
            break
        best = v
    return best


_BATCHED_OBJECTIVES: dict[str, Any] = {}


def _batched_objective(name: str) -> Any:
    if name not in _BATCHED_OBJECTIVES:
        _BATCHED_OBJECTIVES[name] = jax.jit(jax.vmap(family_objective(name)))
    return _BATCHED_OBJECTIVES[name]


def _owned_traces(name: str, contexts: Any, x0: np.ndarray, budget: int) -> list[Any]:
    """Owned L-BFGS (sequential strategy) from per-instance starts; best-so-far traces."""
    spec = FAMILIES[name]
    n = x0.shape[0]
    lo, hi = jnp.tile(spec.lo, (n, 1)), jnp.tile(spec.hi, (n, 1))
    out = lbfgs_solve_batch(family_objective(name), contexts, jnp.asarray(x0), (lo, hi),
                            budget, strategy="map")
    f_starts = np.asarray(_batched_objective(name)(jnp.asarray(x0), contexts))
    traces = []
    for i in range(n):
        k = int(out["accepted"][i])
        f_start = float(f_starts[i])
        units = [2, *[int(u) for u in out["accepted_units"][i][:k]]]
        values = np.minimum.accumulate([f_start if np.isfinite(f_start) else np.inf,
                                        *out["accepted_values"][i][:k]])
        traces.append({"trace": list(zip(units, values.tolist(), strict=True)),
                       "used": int(out["used"][i]), "status": int(out["status"][i]),
                       "objective_evaluations": int(out["objective_evaluations"][i]),
                       "gradient_evaluations": int(out["gradient_evaluations"][i]),
                       "x": np.asarray(out["x"][i]).tolist()})
    return traces


def _nn_features(name: str, contexts: Any) -> np.ndarray:
    """Fixed context representation: observations interpolated to a common grid per
    species (linear, flat beyond the ends), plus the dose."""
    t = np.asarray(contexts["t"])
    y = np.asarray(contexts["y"])
    sp = np.asarray(contexts["species"])
    mask = np.asarray(contexts["mask"])
    dose = np.asarray(contexts["dose"])
    if name == "sinusoid":
        grid, species = 0.25 * np.arange(40), (0,)
    else:
        grid, species = 0.5 * np.arange(1, 21), (0, 1)
    rows = []
    for i in range(t.shape[0]):
        feats = []
        for s in species:
            sel = mask[i] & (sp[i] == s)
            ts, ys = t[i][sel], y[i][sel]
            order = np.argsort(ts)
            feats.append(np.interp(grid, ts[order], ys[order]) if sel.any()
                         else np.zeros_like(grid))
        rows.append(np.concatenate([*feats, [dose[i]]]))
    return np.asarray(rows)


def _fisher_condition(name: str, x: np.ndarray, c: Any) -> float:
    spec = FAMILIES[name]

    def residual(theta: Any) -> Any:
        model = spec.observe(theta, c)
        return jnp.where(c["mask"], (model - c["y"]) / c["sigma"], 0.0)

    j = np.asarray(jax.jacfwd(residual)(jnp.asarray(x)))
    eig = np.linalg.eigvalsh(j.T @ j)
    return float(eig[-1] / max(eig[0], 1e-300))


def _scipy_argmin(runner: Any, starts: list[Any], c: Any) -> np.ndarray:
    """Best final point of precise SciPy L-BFGS-B runs (identifiability check only)."""
    from scipy.optimize import minimize as scipy_minimize

    spec = runner.spec

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        v, g = runner.vg(jnp.asarray(x), c)
        return float(v), np.asarray(g, float)

    best_value, best_x = np.inf, np.asarray(starts[0], float)
    for x0 in starts:
        res = scipy_minimize(fun, np.clip(np.asarray(x0, float), spec.lo, spec.hi), jac=True,
                             method="L-BFGS-B", bounds=list(zip(spec.lo, spec.hi, strict=True)),
                             options={"ftol": 1e-15, "gtol": 1e-10, "maxfun": 1000})
        if np.isfinite(res.fun) and res.fun < best_value:
            best_value, best_x = float(res.fun), np.asarray(res.x)
    return best_x


def _b95(rates: dict[int, float]) -> int | None:
    for b in BUDGETS:
        if rates[b] >= 0.95:
            return b
    return None


def _no_progress(stage: str, done: int, total: int) -> None:
    return None


def run_family(name: str, progress: Any = _no_progress) -> dict[str, Any]:
    spec = FAMILIES[name]
    t_start, cpu_start = time.perf_counter(), time.process_time()
    val, val_truth = make_split(name, "validation", N_VALIDATION)
    n = N_VALIDATION
    ctx = [jax.tree_util.tree_map(lambda a, i=i: a[i], val) for i in range(n)]
    scipy = _ScipyRunner(name)

    # reference: SciPy multistart (64 Sobol + truth), two scramble seeds
    ref = {}
    distinct = []
    for tag in ("reference_a", "reference_b"):
        starts = _sobol(spec, REFERENCE_STARTS, SOBOL_SEEDS[tag])
        best = np.full(n, np.inf)
        for i in range(n):
            progress(tag, i, n)
            finals = []
            for x0 in [*starts, np.asarray(val_truth[i])]:
                tr = scipy.trace(x0, ctx[i], REFERENCE_UNITS, precise=True)
                if tr:
                    finals.append(tr[-1][1])
            best[i] = min(finals)
            if tag == "reference_a":
                rounded = np.round(np.asarray(finals) / max(abs(best[i]), 1e-12), 4)
                distinct.append(len(np.unique(rounded[np.isfinite(rounded)])))
        ref[tag] = best
    f_star = np.minimum(ref["reference_a"], ref["reference_b"])
    reproduced = np.mean([abs(a - b) <= EPS_ABS + EPS_REL * abs(min(a, b))
                          for a, b in zip(ref["reference_a"], ref["reference_b"], strict=True)])

    # arms on validation (one long run per start; budgets read off the traces)
    progress("arms A0/A1", 0, n)
    default = np.tile(spec.default, (n, 1))
    a0 = _owned_traces(name, val, default, LONG)
    a1 = [scipy.trace(spec.default, ctx[i], LONG) for i in range(n)]
    arm_starts = _sobol(spec, max(MULTISTART_K), SOBOL_SEEDS["arms"])
    progress("arms A2/A2o", 0, n)
    a2_traces = [[scipy.trace(arm_starts[k], ctx[i], LONG) for k in range(len(arm_starts))]
                 for i in range(n)]
    a2o_traces = [
        _owned_traces(name, val, np.tile(arm_starts[k], (n, 1)), LONG)
        for k in range(len(arm_starts))
    ]  # [start][instance]

    # training targets for the nearest-neighbour start (owned multistart, label budget)
    train, _ = make_split(name, "train", N_TRAIN)
    label_starts = _sobol(spec, LABEL_STARTS, SOBOL_SEEDS["arms"] + 10)
    best_val = np.full(N_TRAIN, np.inf)
    best_x = np.zeros((N_TRAIN, spec.dimension))
    label_units = 0
    for k in range(LABEL_STARTS):
        progress("training labels (starts)", k, LABEL_STARTS)
        runs = _owned_traces(name, train, np.tile(label_starts[k], (N_TRAIN, 1)), LABEL_UNITS)
        for i, r in enumerate(runs):
            label_units += r["used"]
            if r["trace"][-1][1] < best_val[i]:
                best_val[i], best_x[i] = r["trace"][-1][1], r["x"]
    train_feats = _nn_features(name, train)
    mean, std = train_feats.mean(0), train_feats.std(0) + 1e-12
    z_train = (train_feats - mean) / std
    z_val = (_nn_features(name, val) - mean) / std
    nearest = np.argmin(((z_val[:, None, :] - z_train[None, :, :]) ** 2).sum(-1), axis=1)
    a3 = _owned_traces(name, val, best_x[nearest], LONG)

    # label-reference adequacy: the label procedure on validation versus f*
    label_val_best = np.full(n, np.inf)
    for k in range(LABEL_STARTS):
        progress("label adequacy (starts)", k, LABEL_STARTS)
        for i, r in enumerate(_owned_traces(name, val, np.tile(label_starts[k], (n, 1)),
                                            LABEL_UNITS)):
            label_val_best[i] = min(label_val_best[i], r["trace"][-1][1])

    # any evaluated value below f* updates it (reported)
    arm_best = np.array([min([a0[i]["trace"][-1][1], a1[i][-1][1] if a1[i] else np.inf,
                              a3[i]["trace"][-1][1],
                              *[t[-1][1] for t in a2_traces[i] if t],
                              *[a2o_traces[k][i]["trace"][-1][1] for k in range(len(arm_starts))]])
                         for i in range(n)])
    updated = arm_best < f_star - (EPS_ABS + EPS_REL * np.abs(f_star))
    f_star = np.minimum(f_star, arm_best)
    labels_adequate = float(np.mean([_success(v, fs) for v, fs in zip(label_val_best, f_star,
                                                                      strict=True)]))

    def rates_single(traces: list[Any], key: str | None = "trace") -> dict[int, float]:
        out = {}
        for b in BUDGETS:
            hits = [_success(_best_within(t[key] if key else t, b), f_star[i])
                    for i, t in enumerate(traces)]
            out[b] = float(np.mean(hits))
        return out

    def rates_multi(per_instance_traces: Any) -> tuple[dict[int, float], dict[int, int]]:
        out, chosen = {}, {}
        for b in BUDGETS:
            best_rate, best_k = -1.0, None
            for k in MULTISTART_K:
                per = b // k
                if per < 2:
                    continue
                hits = [_success(min(_best_within(per_instance_traces(i, s), per)
                                     for s in range(k)), f_star[i]) for i in range(n)]
                rate = float(np.mean(hits))
                if rate > best_rate:
                    best_rate, best_k = rate, k
            out[b], chosen[b] = best_rate, best_k
        return out, chosen

    rates = {
        "A0_default_owned": rates_single(a0),
        "A1_default_scipy": rates_single(a1, key=None),
        "A3_nearest_neighbour_owned": rates_single(a3),
    }
    rates["A2_multistart_scipy"], k_a2 = rates_multi(lambda i, s: a2_traces[i][s])
    rates["A2o_multistart_owned"], k_a2o = rates_multi(lambda i, s: a2o_traces[s][i]["trace"])
    b95 = {arm: _b95(r) for arm, r in rates.items()}
    c_local = float(np.median([r["used"] for r in a0]))
    multi = [b for b in (b95["A2_multistart_scipy"], b95["A2o_multistart_owned"]) if b]
    b95_multi = min(multi) if multi else None
    stop = b95_multi is not None and b95_multi <= 2 * c_local
    nn_amortizes = b95["A3_nearest_neighbour_owned"] is not None and \
        b95["A3_nearest_neighbour_owned"] <= 2 * c_local
    hard = np.mean([not any([
        _success(_best_within(a0[i]["trace"], LONG), f_star[i]),
        _success(_best_within(a1[i], LONG), f_star[i]),
        _success(_best_within(a3[i]["trace"], LONG), f_star[i]),
        any(_success(_best_within(t, LONG), f_star[i]) for t in a2_traces[i]),
    ]) for i in range(n)])

    # identifiability at the reference optimum (precise SciPy multistart, first 50 instances)
    starts_a = _sobol(spec, REFERENCE_STARTS, SOBOL_SEEDS["reference_a"])
    progress("identifiability", 0, min(n, 50))
    cond = [_fisher_condition(name, _scipy_argmin(scipy, [*starts_a[:16], val_truth[i]],
                                                   ctx[i]), ctx[i]) for i in range(min(n, 50))]
    cond = np.asarray(cond)

    def units_split(traces: list[Any]) -> dict[str, float]:
        return {"objective_evaluations_mean": float(np.mean([t["objective_evaluations"]
                                                             for t in traces])),
                "gradient_evaluations_mean": float(np.mean([t["gradient_evaluations"]
                                                            for t in traces]))}

    return {
        "family": name,
        "validation_instances": n,
        "success_tolerance": {"eps_abs": EPS_ABS, "eps_rel": EPS_REL},
        "reference": {
            "method": "SciPy L-BFGS-B, 64 Sobol starts + truth, cap 2000 units each",
            "two_seed_reproduction_rate": float(reproduced),
            "updated_by_arms": int(updated.sum()),
            "f_star_median": float(np.median(f_star)),
            "distinct_optima_per_instance": {
                "median": float(np.median(distinct)), "max": int(np.max(distinct)),
                "share_with_more_than_one": float(np.mean(np.asarray(distinct) > 1))},
        },
        "training_targets": {"instances": N_TRAIN, "starts": LABEL_STARTS,
                             "cap_units": LABEL_UNITS, "total_units": int(label_units),
                             "validation_adequacy_rate": labels_adequate},
        "success_rate_by_budget": rates,
        "multistart_k_chosen": {"A2": k_a2, "A2o": k_a2o},
        "B95": b95,
        "c_local_median_units": c_local,
        "B95_multi_better_of_A2_A2o": b95_multi,
        "stop_rule": {"rule": "stop if B95_multi <= 2 * c_local", "stop": bool(stop),
                      "nn_amortizes": bool(nn_amortizes)},
        "hard_set_share_at_800": float(hard),
        "calls_at_800": {"A0": units_split(a0), "A3": units_split(a3)},
        "identifiability": {
            "instances_checked": int(len(cond)), "cond_limit": FISHER_COND_LIMIT,
            "share_non_identified": float(np.mean(cond > FISHER_COND_LIMIT)),
            "cond_median": float(np.median(cond)), "cond_max": float(np.max(cond)),
        },
        "scipy_calls": int(scipy.calls),
        "wall_seconds": time.perf_counter() - t_start,
        "cpu_seconds": time.process_time() - cpu_start,
    }


def split_manifest() -> dict[str, Any]:
    from neuralopt.families import split_digest

    sizes = {"train": N_TRAIN, "validation": N_VALIDATION, "test": 200}
    out = {}
    for name in VARIANTS:
        out[name] = {}
        for split in FAMILIES[name].splits:
            count = sizes.get(split, 200)
            out[name][split] = {"instances": count,
                                "sha256": split_digest(*make_split(name, split, count))}
    return out


RUN_FILES = (ROOT / "scripts" / "run_f0_v0_8.py", ROOT / "scripts" / "run_checkpoint.py",
             ROOT / "src" / "neuralopt" / "families.py", ROOT / "src" / "neuralopt" / "lbfgs.py",
             ROOT / "uv.lock")


def _run_report() -> dict[str, Any]:
    env = {"jax": jax.__version__, "python": platform.python_version(),
           "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "xla_flags": os.environ.get("XLA_FLAGS"), "float": "float64"}
    return {"schema": "neuralopt-f0/v0.8", "environment": env,
            "protocol": {"budgets": BUDGETS, "multistart_k": MULTISTART_K,
                         "sobol_seeds": SOBOL_SEEDS, "eps_abs": EPS_ABS,
                         "eps_rel": EPS_REL, "fisher_cond_limit": FISHER_COND_LIMIT,
                         "accounting": "objective and gradient calls; SciPy call = "
                         "objective + gradient (2 units); owned: objective per trial, "
                         "gradient on acceptance",
                         "scipy_best_so_far": "over all evaluated points (as in v0.7)"},
            "splits": split_manifest(), "families": []}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run_in_directory(run_dir: Path, families: list[str]) -> int:
    """F0 with per-family checkpoints in ``run_dir``; rerunning resumes."""
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "f0-v0.8.0.json"
    if final.exists():
        raise SystemExit(f"{final} exists; results are never overwritten")
    code = run_checkpoint.fingerprint(RUN_FILES)
    lock = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    with (run_dir / "command.txt").open("a") as handle:
        handle.write(f"start_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                     f"command: {' '.join([sys.executable, *sys.argv])}\n"
                     f"git_sha: {_git('rev-parse', 'HEAD')}\n"
                     f"git_dirty: {bool(_git('status', '--porcelain', '--untracked-files=no'))}\n"
                     f"uv_lock_sha256: {lock}\ncode_fingerprint: {code}\n\n")
    probe = ROOT / "results" / "f0-probe-v0.8.0.json"
    weights = {r["family"]: r["projection"]["total_cpu_hours"]
               for r in json.loads(probe.read_text())["rows"]} if probe.exists() else None

    def compute(name: str, progress: Any) -> dict[str, Any]:
        row = run_family(name, progress)
        print(json.dumps({k: row[k] for k in ("family", "B95", "c_local_median_units",
                                              "B95_multi_better_of_A2_A2o", "stop_rule",
                                              "wall_seconds")}), flush=True)
        return row

    rows = run_checkpoint.run_units(families, run_dir, compute, fingerprint=code,
                                    weights=weights)
    report = _run_report()
    report["code_fingerprint"] = code
    report["families"] = rows
    run_checkpoint.write_atomic(final, json.dumps(report, indent=2) + "\n")
    (run_dir / "DONE").write_text(f"{final.name}\n")
    print(f"F0 complete: {final}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["probe", "run"])
    parser.add_argument("--families", nargs="*", default=list(VARIANTS))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "f0-probe-v0.8.0.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="run directory (run only): per-family checkpoints, progress.json, "
                        "command.txt, final report; rerun with the same directory to resume")
    args = parser.parse_args()
    if args.command == "run" and args.out is not None:
        return run_in_directory(args.out, list(args.families))
    if args.command == "run" and args.output.name == "f0-probe-v0.8.0.json":
        args.output = ROOT / "results" / "f0-v0.8.0.json"
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; results are never overwritten")
    if args.command == "run":
        report = _run_report()
        for name in args.families:
            row = run_family(name)
            report["families"].append(row)
            print(json.dumps({k: row[k] for k in ("family", "B95", "c_local_median_units",
                                                  "B95_multi_better_of_A2_A2o", "stop_rule",
                                                  "wall_seconds")}), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0
    rows = []
    for name in VARIANTS:
        row = probe_family(name)
        row["projection"] = project(row)
        rows.append(row)
        print(json.dumps(row, indent=1))
    total = sum(r["projection"]["total_cpu_hours"] for r in rows)
    report = {
        "schema": "neuralopt-f0-probe/v0.8",
        "environment": {"jax": jax.__version__, "python": platform.python_version(),
                        "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "xla_flags": os.environ.get("XLA_FLAGS"), "float": "float64"},
        "protocol": {"validation_instances": N_VALIDATION, "train_instances": N_TRAIN,
                     "reference": f"2 seeds x {REFERENCE_STARTS}+1 starts, cap {REFERENCE_UNITS}",
                     "labels": f"{LABEL_STARTS} starts, cap {LABEL_UNITS}",
                     "budgets": BUDGETS, "cap_cpu_hours": CAP_CPU_HOURS},
        "rows": rows,
        "projected_total_cpu_hours": total,
        "within_cap": bool(total <= CAP_CPU_HOURS),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"projected F0 total: {total:.2f} CPU-hours (cap {CAP_CPU_HOURS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
