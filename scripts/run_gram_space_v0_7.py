"""v0.7 Track B: training, evaluation and statistics for the frozen pre-registration.

Pre-registration: docs/superpowers/specs/2026-11-22-gram-space-v07-preregistration.md
(its sha256 is checked at startup). Every constant below is copied from it. Classical
references (BB = τ-only, the BFGS oracle, SciPy L-BFGS-B) run only here; NeuralOpt
inference never contains them.

Usage::

    uv run python scripts/run_gram_space_v0_7.py --probe        # parity check + timing probe
    uv run python scripts/run_gram_space_v0_7.py --stage 1      # train gram-scalar, gram-nohist
    uv run python scripts/run_gram_space_v0_7.py --evaluate 1   # build test tasks; S1 decision
    uv run python scripts/run_gram_space_v0_7.py --stage 2      # only if S1 passes
    uv run python scripts/run_gram_space_v0_7.py --evaluate 2
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ["JAX_ENABLE_X64"] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import cache  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from neuralopt.compiled import preconditioned_solve, preconditioned_solve_batch  # noqa: E402
from neuralopt.fast_training import train_batched  # noqa: E402
from neuralopt.features import gradient_to_normalized, normalize_coordinates  # noqa: E402
from neuralopt.gram import (  # noqa: E402
    HISTORY_V2_MASKS,
    GramConfig,
    RejectionInfo,
    apply_inverse_metric,
    gram_basis,
    gram_step,
    init_gram_model,
    load_gram_checkpoint,
    save_gram_checkpoint,
    tau_scale,
)
from neuralopt.gram_training import GramUnrollConfig  # noqa: E402
from neuralopt.loop_rules import accept_step  # noqa: E402
from neuralopt.task_families import family_objective, make_task, stack_tasks  # noqa: E402
from neuralopt.trajectory import empty_history, push_history, record_rejection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gram_gates_v0_7 import oracle_coefficients  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "docs/superpowers/specs/2026-11-22-gram-space-v07-preregistration.md"
PREREG_SHA = PREREG.with_suffix(".sha256")
VERSION = "0.7.0"

# ---- frozen constants (pre-registration sections 4-8) -------------------------------
BASE = dict(history_length=10, width=32, heads=4, layers=2, **HISTORY_V2_MASKS)
ARMS = {
    "gram-scalar": GramConfig(**BASE, scalar_only=True),
    "gram-nohist": GramConfig(**BASE, history_masked=True),
    "gram-full": GramConfig(**BASE),
}
STAGE_ARMS = {1: ("gram-scalar", "gram-nohist"), 2: ("gram-full",)}
REFERENCE_CONFIG = GramConfig(**BASE)
SEEDS, UPDATES, BATCH, LEARNING_RATE = (0, 1, 2, 3, 4), 2000, 8, 1e-3
UNROLL = GramUnrollConfig(budget=100, horizons=(25, 50, 100), candidate_weight=0.1,
                          regret_floor=1e-12)
TRAIN_VARIANTS = (("isotropic", 1.0), ("diagonal", 10.0), ("diagonal", 100.0),
                  ("diagonal", 1000.0), ("rotated", 10.0), ("rotated", 100.0),
                  ("rotated", 1000.0), ("rastrigin", 0.0))
TRAIN_DIMS, TEST_DIMS, PER_CELL = (4, 8, 12), (7, 16, 32), 4
TEST_FAMILIES = (("diagonal", 1000.0), ("rotated", 100.0), ("rotated", 1000.0),
                 ("isotropic", 1.0), ("rastrigin", 0.0), ("ackley", 0.0),
                 ("rosenbrock-shifted", 0.0))
CURVATURE = ("diagonal-1000", "rotated-100", "rotated-1000")
GUARD = ("rastrigin", "ackley", "rosenbrock-shifted")
BUDGETS, PRIMARY_BUDGET = (50, 100, 200), 100
LOG_FLOOR, RAW_FLOOR, DRAWS = -12.0, 1e-16, 10_000
S1_MARGIN, NI_MARGIN, EQUIVALENCE, GUARD_MEAN = 0.25, 0.5, 0.5, 0.5
RELEARNED_GAP, RELEARNED_FRACTION, PROBE_STATES, PROBE_COSINE = 0.25, 0.75, 200, 0.99
PROBE_UPDATES, PROBE_LIMIT_MINUTES = 20, 45.0
TF_H8 = [ROOT / "artifacts" / f"trajectory-v0.6.0-tf-h8-seed{s}.npz" for s in (0, 1, 2)]


def check_preregistration() -> str:
    digest = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if digest != PREREG_SHA.read_text().split()[0]:
        raise RuntimeError("pre-registration file changed after freezing")
    return digest


# ---- tasks ---------------------------------------------------------------------------
def _scales(index: int) -> tuple[float, float]:
    return (0.2, 1.0, 5.0)[index % 3], (0.5, 1.0, 2.0)[(2 * index + 1) % 3]


def training_specs() -> list[dict[str, Any]]:
    specs = []
    for i, (family, condition) in enumerate(TRAIN_VARIANTS):
        for j, dim in enumerate(TRAIN_DIMS):
            for r in range(PER_CELL):
                a, c = _scales(i + j + r)
                specs.append(dict(family=family, condition=condition, dimension=dim,
                                  seed=7000 + 100 * i + 10 * j + r, objective_scale=a,
                                  coordinate_scale=c, replicate=r))
    return specs


def test_specs() -> list[dict[str, Any]]:
    specs = []
    for i, (family, condition) in enumerate(TEST_FAMILIES):
        for j, dim in enumerate(TEST_DIMS):
            for r in range(PER_CELL):
                # Equal-spectrum pairs: diagonal and rotated kappa=1000 share seeds and
                # scales (same permuted spectrum; the rotated one adds Q).
                base = 2 if (family, condition) == ("diagonal", 1000.0) else i
                a, c = _scales(base + j + r)
                specs.append(dict(family=family, condition=condition, dimension=dim,
                                  seed=9000 + 100 * base + 10 * j + r, objective_scale=a,
                                  coordinate_scale=c, replicate=r))
    return specs


def split_hash(specs: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()


def build(spec: dict[str, Any]) -> Any:
    return make_task(spec["family"], spec["condition"], spec["dimension"], spec["seed"],
                     spec["objective_scale"], spec["coordinate_scale"])


def training_groups() -> list[Any]:
    """Groups by (objective family, dimension), cycled in proportion to their sizes
    (pre-training implementation detail; recorded under Corrections)."""
    groups = stack_tasks([build(s) for s in training_specs()])
    smallest = min(g.size for g in groups)
    return [g for g in groups for _ in range(g.size // smallest)]


# ---- training ------------------------------------------------------------------------
def checkpoint_path(arm: str, seed: int) -> Path:
    return ROOT / "artifacts" / f"gram-v{VERSION}-{arm}-seed{seed}.npz"


def train_arm(job: dict[str, Any]) -> dict[str, Any]:
    arm, seed, updates = job["arm"], job["seed"], job["updates"]
    start = time.perf_counter()
    report = train_batched(training_groups(), seed=seed, steps=updates, batch_size=BATCH,
                           learning_rate=LEARNING_RATE, config=ARMS[arm], unroll=UNROLL,
                           model_kind="gram")
    info = {
        "arm": arm, "seed": seed, "updates": updates,
        "wall_seconds": time.perf_counter() - start, "update_seconds": report.seconds,
        "skipped_updates": report.skipped_updates, "skipped_steps": report.skipped_steps,
        "first_loss": report.losses[0], "mean_last_100_loss": float(np.mean(report.losses[-100:])),
        "parameter_count": int(sum(np.asarray(v).size for v in report.params.values())),
        "train_split_sha256": split_hash(training_specs()),
        "preregistration_sha256": job["prereg"], "rng": f"jax.random.PRNGKey({seed})",
    }
    if job.get("save", True):
        path = checkpoint_path(arm, seed)
        save_gram_checkpoint(path, report.params, ARMS[arm], info)
        info["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        info["checkpoint_bytes"] = path.stat().st_size
    return info


def timing_probe() -> dict[str, Any]:
    """20 updates of gram-full, seed 0; nothing is saved."""
    prereg = check_preregistration()
    groups = training_groups()
    distinct = len({g.name for g in groups})
    start = time.perf_counter()
    first = train_batched(groups, seed=0, steps=distinct, batch_size=BATCH,
                          learning_rate=LEARNING_RATE, config=ARMS["gram-full"], unroll=UNROLL,
                          model_kind="gram")
    compile_pass = time.perf_counter() - start
    report = train_batched(groups, seed=0, steps=PROBE_UPDATES, batch_size=BATCH,
                           learning_rate=LEARNING_RATE, config=ARMS["gram-full"],
                           unroll=UNROLL, model_kind="gram")
    per_update = report.seconds / PROBE_UPDATES
    tasks_per_update = float(np.mean([min(BATCH, groups[s % len(groups)].size)
                                      for s in range(PROBE_UPDATES)]))
    run_minutes = (compile_pass + per_update * UPDATES) / 60
    runs = sum(len(a) for a in STAGE_ARMS.values()) * len(SEEDS)
    return {
        "preregistration_sha256": prereg, "distinct_compilations": distinct,
        "first_pass_seconds_incl_compile": compile_pass,
        "steady_seconds_per_update": per_update,
        "steady_task_gradients_per_second": tasks_per_update / per_update,
        "skipped_updates_in_probe": report.skipped_updates + first.skipped_updates,
        "projected_minutes_per_run_single_process": run_minutes, "runs": runs,
        "projected_cpu_hours_total": runs * run_minutes / 60,
        "probe_rule": f"lower UPDATES uniformly only if a run exceeds {PROBE_LIMIT_MINUTES} min",
        "updates_kept": UPDATES if run_minutes <= PROBE_LIMIT_MINUTES else None,
    }


# ---- references (harness only) ---------------------------------------------------------------
@cache
def reference_loop(source: str, family: str, budget: int, collect: int = 0) -> Any:
    """Frozen loop (halving) with history-v2 masks. ``source``: "bb" (A = C = 0, δ = 0,
    σ = 1) or "oracle" (BFGS coefficients, σ = 1). ``collect`` > 0 records that many
    loop states for the probe-direction criterion."""
    config = REFERENCE_CONFIG
    objective = family_objective(family)

    @jax.jit
    def run(theta: Any, x0: Any, lo: Any, hi: Any) -> dict[str, Any]:
        def f(x: Any) -> Any:
            return objective(x, theta)

        grad_f = jax.grad(f)
        value, grad = f(x0), grad_f(x0)
        hist = push_history(empty_history(config.history_length, x0.shape[0], x0.dtype),
                            normalize_coordinates(x0, lo, hi),
                            gradient_to_normalized(grad, lo, hi), value)
        i32, zero = jnp.int32, jnp.asarray(0.0)
        state = dict(x=x0, value=value, grad=grad, hist=hist, shrink=jnp.asarray(1.0),
                     used=jnp.asarray(2, i32), objective=jnp.asarray(1, i32),
                     gradient=jnp.asarray(1, i32), iterations=jnp.asarray(0, i32),
                     accepted=jnp.asarray(0, i32), initial=jnp.asarray(0, i32),
                     rl=zero, rr=zero, rh=jnp.asarray(False))
        if collect:
            state["states"] = dict(
                hist=jax.tree_util.tree_map(lambda a: jnp.zeros((collect, *a.shape), a.dtype),
                                            hist),
                g_u=jnp.zeros((collect, x0.shape[0])), used=jnp.zeros((collect,), i32),
                rl=jnp.zeros((collect,)), rr=jnp.zeros((collect,)),
                rh=jnp.zeros((collect,), bool), oracle=jnp.zeros((collect, x0.shape[0])))

        def keep_going(s: dict[str, Any]) -> Any:
            g_u = gradient_to_normalized(s["grad"], lo, hi)
            return (s["used"] + 2 <= budget) & (jnp.linalg.norm(g_u) != 0)

        def body(s: dict[str, Any]) -> dict[str, Any]:
            g_u = gradient_to_normalized(s["grad"], lo, hi)
            basis = gram_basis(s["hist"], config)
            tau = tau_scale(basis, g_u, config.fallback_length)
            zeros = jnp.zeros((config.history_length,) * 2)
            if source == "oracle":
                A, C, delta, _ = oracle_coefficients(basis, g_u, tau)
            else:
                A, C, delta = zeros, zeros, jnp.asarray(0.0)
            base = apply_inverse_metric(g_u, basis, A, C, delta, tau, config.eps_rel)
            out = accept_step("halving", f, grad_f, s["x"], s["value"], s["grad"], base,
                              jnp.asarray(1.0), s["shrink"], budget - s["used"], lo, hi)
            pushed = push_history(s["hist"], normalize_coordinates(out.x, lo, hi),
                                  gradient_to_normalized(out.grad, lo, hi), out.value)
            new_hist = jax.tree_util.tree_map(lambda a, b: jnp.where(out.accepted, a, b),
                                              pushed, record_rejection(s["hist"]))
            u = normalize_coordinates(s["x"], lo, hi)
            step_u = normalize_coordinates(out.trial_x, lo, hi) - u
            predicted = jnp.sum(g_u * step_u)
            ratio = jnp.where(jnp.isfinite(out.trial_value) & (predicted != 0),
                              (out.trial_value - s["value"])
                              / jnp.where(predicted != 0, predicted, 1.0), 10.0)
            i = s["iterations"]
            new = dict(
                x=out.x, value=out.value, grad=out.grad, hist=new_hist,
                shrink=out.next_shrink.astype(x0.dtype),
                used=s["used"] + out.objective_evaluations + out.gradient_evaluations,
                objective=s["objective"] + out.objective_evaluations,
                gradient=s["gradient"] + out.gradient_evaluations, iterations=i + 1,
                accepted=s["accepted"] + out.accepted.astype(i32),
                initial=s["initial"] + out.initial_accepted.astype(i32),
                rl=jnp.where(out.accepted, 0.0, jnp.linalg.norm(step_u)),
                rr=jnp.where(out.accepted, 0.0, jnp.clip(ratio, -10.0, 10.0)),
                rh=~out.accepted,
            )
            if collect:
                slot = jnp.minimum(i, collect - 1)
                st = s["states"]
                new["states"] = dict(
                    hist=jax.tree_util.tree_map(lambda a, b: a.at[slot].set(b), st["hist"],
                                                s["hist"]),
                    g_u=st["g_u"].at[slot].set(g_u), used=st["used"].at[slot].set(s["used"]),
                    rl=st["rl"].at[slot].set(s["rl"]), rr=st["rr"].at[slot].set(s["rr"]),
                    rh=st["rh"].at[slot].set(s["rh"]), oracle=st["oracle"].at[slot].set(base))
            return new

        return jax.lax.while_loop(keep_going, body, state)

    return run


def lbfgsb(task: Any, budget: int) -> tuple[float, int]:
    from scipy.optimize import minimize as scipy_minimize

    value_and_grad = _value_and_grad(task.family)
    calls, best = 0, np.inf

    class Exhausted(Exception):
        pass

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls, best
        if calls >= budget // 2:  # every call, line-search trials included: 2 units
            raise Exhausted
        calls += 1
        v, g = value_and_grad(x, task.theta)
        if np.isfinite(float(v)):
            best = min(best, float(v))
        return float(v), np.asarray(g, dtype=float)

    try:
        scipy_minimize(fun, np.asarray(task.x0), jac=True, method="L-BFGS-B",
                       bounds=list(zip(np.asarray(task.lo), np.asarray(task.hi), strict=True)),
                       options={"maxfun": budget // 2, "maxiter": budget, "ftol": 0.0,
                                "gtol": 0.0, "maxcor": 10})
    except Exhausted:
        pass
    return best, 2 * calls


@cache
def _value_and_grad(family: str) -> Any:
    return jax.jit(jax.value_and_grad(family_objective(family)))


def reference_parity_check() -> bool:
    """With all-zero Gram parameters (σ = 1, δ = 0, A = C = 0) the package's compiled
    Gram loop is BB; the harness "bb" loop must reproduce it to 1e-10."""
    task = make_task("rotated", 100.0, 7, 5996)
    params = {k: jnp.zeros_like(v) for k, v in init_gram_model(jax.random.PRNGKey(0),
                                                               REFERENCE_CONFIG).items()}
    package = preconditioned_solve(family_objective(task.family), task.x0, (task.lo, task.hi),
                                   60, theta=task.theta, params=params,
                                   config=REFERENCE_CONFIG, model_kind="gram")
    ours = reference_loop("bb", task.family, 60)(task.theta, task.x0, task.lo, task.hi)
    return bool(np.isclose(float(ours["value"]), float(package["value"]), rtol=1e-10)
                and int(ours["used"]) == int(package["used"]))


# ---- evaluation ------------------------------------------------------------------------
def _log10_regret(value: float, f0: float, optimum: float) -> float:
    regret = max(value - optimum, 0.0) / (abs(f0 - optimum) + 1e-12)
    return float(np.log10(regret + RAW_FLOOR)) if np.isfinite(value) else float("nan")


def evaluate_rows(specs: list[dict[str, Any]], learned: dict[str, list[tuple[int, Any]]],
                  budgets: tuple[int, ...] = BUDGETS, include_tf_h8: bool = True,
                  include_lbfgsb: bool = True) -> list[dict[str, Any]]:
    """One row per (arm, training seed or None, task, budget)."""
    tasks = [build(s) for s in specs]
    rows: list[dict[str, Any]] = []

    def add(arm: str, seed: Any, index: int, budget: int, value: float, units: int,
            extra: dict[str, Any] | None = None) -> None:
        task, spec = tasks[index], specs[index]
        f0 = float(family_objective(task.family)(task.x0, task.theta))
        rows.append(dict(arm=arm, seed=seed, task=index, label=task.label,
                         dimension=spec["dimension"], replicate=spec["replicate"],
                         budget=budget, log10_regret=_log10_regret(value, f0, 0.0),
                         units=units, **(extra or {})))

    for budget in budgets:
        for i, task in enumerate(tasks):
            for source in ("bb", "oracle"):
                out = reference_loop(source, task.family, budget)(task.theta, task.x0,
                                                                  task.lo, task.hi)
                add(source, None, i, budget, float(out["value"]), int(out["used"]),
                    {"iterations": int(out["iterations"]), "accepted": int(out["accepted"]),
                     "initial": int(out["initial"])})
            if include_lbfgsb:
                best, units = lbfgsb(task, budget)
                add("lbfgsb", None, i, budget, best, units)
            if include_tf_h8:
                from neuralopt.training import load_checkpoint

                for seed, path in enumerate(TF_H8):
                    params, config = load_checkpoint(path)
                    out = preconditioned_solve(family_objective(task.family), task.x0,
                                               (task.lo, task.hi), budget, theta=task.theta,
                                               params=params, config=config)
                    add("tf-h8", seed, i, budget, float(out["value"]), int(out["used"]))
        for arm, models in learned.items():
            for seed, params in models:
                for group_index in _groups(tasks):
                    group = [tasks[k] for k in group_index]
                    thetas = jax.tree_util.tree_map(lambda *a: jnp.stack(a),
                                                    *[t.theta for t in group])
                    out = preconditioned_solve_batch(
                        family_objective(group[0].family), thetas,
                        jnp.stack([t.x0 for t in group]),
                        (jnp.stack([t.lo for t in group]), jnp.stack([t.hi for t in group])),
                        budget, params=params, config=ARMS[arm], model_kind="gram")
                    for slot, k in enumerate(group_index):
                        units = int(out["objective_evaluations"][slot]
                                    + out["gradient_evaluations"][slot])
                        add(arm, seed, k, budget, float(out["value"][slot]), units,
                            {"iterations": int(out["iteration"][slot]),
                             "accepted": int(out["accepted_steps"][slot]),
                             "initial": int(np.sum(out["initial_accepted"][slot])),
                             "status": int(out["status"][slot])})
    return rows


def _groups(tasks: list[Any]) -> list[list[int]]:
    groups: dict[tuple[str, int], list[int]] = {}
    for k, task in enumerate(tasks):
        groups.setdefault((task.family, task.dimension), []).append(k)
    return list(groups.values())


def probe_cosines(specs: list[dict[str, Any]], params: Any, config: GramConfig,
                  states: int = PROBE_STATES) -> list[float]:
    """Cosine between the model's and the oracle's step directions on fixed oracle states
    (criterion R(ii)); states are taken in task order until ``states`` are collected."""
    cosines: list[float] = []
    for spec in specs:
        task = build(spec)
        run = reference_loop("oracle", task.family, PRIMARY_BUDGET, collect=PRIMARY_BUDGET)
        out = run(task.theta, task.x0, task.lo, task.hi)
        st = out["states"]
        for i in range(int(out["iterations"])):
            hist = jax.tree_util.tree_map(lambda a, i=i: a[i], st["hist"])
            rejection = RejectionInfo(hist.rejections, st["rl"][i], st["rr"][i], st["rh"][i])
            base, sigma, _, _ = gram_step(params, config, hist, st["g_u"][i], rejection,
                                          1 - st["used"][i] / PRIMARY_BUDGET)
            model, oracle = np.asarray(sigma * base), np.asarray(st["oracle"][i])
            denominator = np.linalg.norm(model) * np.linalg.norm(oracle)
            cosines.append(float(model @ oracle / denominator) if denominator > 0 else 0.0)
            if len(cosines) >= states:
                return cosines
    return cosines


# ---- statistics and decisions ---------------------------------------------------------
def paired_interval(matrix: np.ndarray, seed: int = 0) -> dict[str, Any]:
    """Mean of ``matrix[training_seed, task]``; 95% two-level crossed bootstrap
    (a single row = reference-vs-reference: tasks only)."""
    matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("nonfinite paired differences must be resolved explicitly")
    rng = np.random.default_rng(seed)
    n_seeds, n_tasks = matrix.shape
    s = rng.integers(0, n_seeds, (DRAWS, n_seeds))
    t = rng.integers(0, n_tasks, (DRAWS, n_tasks))
    draws = np.array([matrix[a][:, b].mean() for a, b in zip(s, t, strict=True)])
    return {"mean": float(matrix.mean()), "ci95": [float(np.quantile(draws, 0.025)),
                                                   float(np.quantile(draws, 0.975))],
            "n_seeds": int(n_seeds), "n_tasks": int(n_tasks)}


def superiority(interval: dict[str, Any]) -> bool:
    return interval["ci95"][1] < 0


def non_inferior(interval: dict[str, Any], margin: float) -> bool:
    return interval["ci95"][1] < margin


def equivalent(interval: dict[str, Any], margin: float = EQUIVALENCE) -> bool:
    return -margin < interval["ci95"][0] and interval["ci95"][1] < margin


def guard_violated(intervals: dict[str, dict[str, Any]]) -> bool:
    return any(i["mean"] > GUARD_MEAN and i["ci95"][0] > 0 for i in intervals.values())


def difference(rows: list[dict[str, Any]], candidate: str, reference: str, labels: tuple,
               budget: int = PRIMARY_BUDGET) -> np.ndarray:
    """``[seed, task]`` floored differences; a reference with one row is broadcast."""
    def table(arm: str) -> dict[Any, dict[int, float]]:
        out: dict[Any, dict[int, float]] = {}
        for r in rows:
            if r["arm"] == arm and r["budget"] == budget and r["label"] in labels:
                out.setdefault(r["seed"], {})[r["task"]] = max(r["log10_regret"], LOG_FLOOR)
        return out

    cand, ref = table(candidate), table(reference)
    tasks = sorted(next(iter(cand.values())))
    ref_rows = list(ref.values())
    matrix = []
    for k, seed in enumerate(sorted(cand, key=lambda s: -1 if s is None else s)):
        r = ref.get(seed, ref_rows[0] if len(ref_rows) == 1 else ref_rows[k])
        matrix.append([cand[seed][t] - r[t] for t in tasks])
    return np.asarray(matrix)


def rotation_difference(rows: list[dict[str, Any]], arm: str,
                        budget: int = PRIMARY_BUDGET) -> np.ndarray:
    """``[seed, pair]``: rotated κ1000 − diagonal κ1000 at equal spectrum (paired by
    dimension and replicate)."""
    def table(label: str) -> dict[Any, dict[tuple[int, int], float]]:
        out: dict[Any, dict[tuple[int, int], float]] = {}
        for r in rows:
            if r["arm"] == arm and r["budget"] == budget and r["label"] == label:
                out.setdefault(r["seed"], {})[(r["dimension"], r["replicate"])] = max(
                    r["log10_regret"], LOG_FLOOR)
        return out

    rot, diag = table("rotated-1000"), table("diagonal-1000")
    keys = sorted(next(iter(rot.values())))
    return np.asarray([[rot[s][k] - diag[s][k] for k in keys]
                       for s in sorted(rot, key=lambda s: -1 if s is None else s)])


def claim(s1_pass: bool, h1: bool | None = None, guard: bool | None = None,
          h3: bool | None = None, relearned: bool | None = None,
          h2: bool | None = None) -> str:
    """The pre-registered outcome-to-claim table (section 8), as code."""
    if not s1_pass:
        return ("The learned scalar step is worse than BB beyond the 0.25-decade margin; "
                "the round ends; BB stays the bar; no Stage 2.")
    if not h1:
        return ("Learned geometry did not beat BB on the curvature domain; BB stays the bar; "
                "H2 and H3 are exploratory, no claim.")
    if guard:
        return ("No superiority claim: the curvature-domain gain comes with a reliability "
                "regression on a multimodal family; both exploratory.")
    rotation = ("no measurable rotation penalty (±0.5)" if h2
                else "no rotation-invariance claim; the rotated−diagonal gap is reported")
    if not h3:
        return (f"Beats BB on unseen rotated/ill-conditioned tasks at equal budget; {rotation}; "
                "short of the oracle-BFGS ceiling.")
    if relearned:
        return (f"Beats BB; {rotation}; by relearning L-BFGS-like directions in this loop: "
                "no evidence of a new method.")
    return (f"Beats BB; {rotation}; non-inferior to oracle-BFGS (+0.5) without matching its "
            "directions; whether it differs usefully from L-BFGS is not established.")


def stage1_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s1 = paired_interval(difference(rows, "gram-scalar", "bb", CURVATURE))
    return {
        "S1_gram_scalar_minus_bb": s1,
        "S1_pass": non_inferior(s1, S1_MARGIN),
        "S1b_gram_scalar_minus_gram_nohist": paired_interval(
            difference(rows, "gram-scalar", "gram-nohist", CURVATURE)),
        "S1b_gram_nohist_minus_bb": paired_interval(
            difference(rows, "gram-nohist", "bb", CURVATURE)),
    }


def stage2_decision(rows: list[dict[str, Any]], cosines: list[float]) -> dict[str, Any]:
    h1 = paired_interval(difference(rows, "gram-full", "bb", CURVATURE))
    guard = {fam: paired_interval(difference(rows, "gram-full", "bb", (fam,))) for fam in GUARD}
    h2 = paired_interval(rotation_difference(rows, "gram-full"))
    h3 = paired_interval(difference(rows, "gram-full", "oracle", CURVATURE))
    gaps = np.abs(difference(rows, "gram-full", "oracle", CURVATURE))
    r_fraction = float(np.mean(gaps <= RELEARNED_GAP))
    r_cosine = float(np.median(cosines)) if cosines else float("nan")
    relearned = bool(non_inferior(h3, NI_MARGIN) and r_fraction >= RELEARNED_FRACTION
                     and r_cosine >= PROBE_COSINE)
    context = {arm: paired_interval(rotation_difference(rows, arm))
               for arm in ("bb", "oracle", "lbfgsb", "tf-h8")}
    decision = {
        "H1_gram_full_minus_bb": h1, "H1": superiority(h1),
        "guard_gram_full_minus_bb": guard, "guard_violated": guard_violated(guard),
        "H2_rotated_minus_diagonal": h2, "H2": equivalent(h2),
        "H2_context": context,
        "H3_gram_full_minus_oracle": h3, "H3": non_inferior(h3, NI_MARGIN),
        "R_fraction_within_0.25": r_fraction, "R_median_probe_cosine": r_cosine,
        "R": relearned,
        "improves_on_oracle": superiority(h3),
    }
    decision["claim"] = claim(True, decision["H1"], decision["guard_violated"],
                              decision["H3"], decision["R"], decision["H2"])
    return decision


# ---- orchestration ----------------------------------------------------------------------
def load_arm(arm: str) -> list[tuple[int, Any]]:
    out = []
    for seed in SEEDS:
        params, config = load_gram_checkpoint(checkpoint_path(arm, seed))
        if config != ARMS[arm]:
            raise ValueError(f"{arm} seed {seed}: checkpoint config differs from the frozen one")
        out.append((seed, params))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--stage", type=int, choices=(1, 2))
    parser.add_argument("--evaluate", type=int, choices=(1, 2))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    prereg = check_preregistration()
    if args.probe:
        result = {"reference_parity": reference_parity_check(), **timing_probe()}
        (ROOT / "results" / f"gram-probe-v{VERSION}.json").write_text(json.dumps(result,
                                                                                indent=1))
        print(json.dumps(result, indent=1))
        return
    if args.stage:
        jobs = [{"arm": arm, "seed": seed, "updates": UPDATES, "prereg": prereg}
                for arm in STAGE_ARMS[args.stage] for seed in SEEDS]
        with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
            training = list(pool.map(train_arm, jobs))
        (ROOT / "results" / f"gram-training-stage{args.stage}-v{VERSION}.json").write_text(
            json.dumps(training, indent=1))
        return
    if args.evaluate:
        assert reference_parity_check()
        specs = test_specs()  # built only now, after training
        arms = [a for stage in range(1, args.evaluate + 1) for a in STAGE_ARMS[stage]]
        rows = evaluate_rows(specs, {arm: load_arm(arm) for arm in arms})
        output: dict[str, Any] = {
            "schema": "neuralopt-gram-space-results/v1", "preregistration_sha256": prereg,
            "test_split_sha256": split_hash(specs),
            "train_split_sha256": split_hash(training_specs()),
            "stage1": stage1_decision(rows), "rows": rows,
        }
        if args.evaluate == 2:
            if not output["stage1"]["S1_pass"]:
                raise SystemExit("Stage 1 failed its stop rule; Stage 2 is not evaluated")
            curvature = [s for s in specs if build(s).label in CURVATURE][:12]
            cosines = probe_cosines(curvature, load_arm("gram-full")[0][1], ARMS["gram-full"])
            output["stage2"] = stage2_decision(rows, cosines)
        else:
            output["stage1"]["claim_if_stopped"] = None if output["stage1"]["S1_pass"] else claim(
                False)
        path = ROOT / "results" / f"gram-space-stage{args.evaluate}-v{VERSION}.json"
        path.write_text(json.dumps(output, indent=1))
        print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
