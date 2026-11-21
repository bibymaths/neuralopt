"""Pre-registered v0.6 trajectory-curvature ablation (see the v0.6 design document).

Protocol constants below are frozen by
``docs/superpowers/specs/2026-11-21-trajectory-curvature-v06-design.md``. Classical
references (BB step, SciPy L-BFGS-B) live only in this harness; NeuralOpt inference never
calls them. Final-test tasks are built only after training and validation tuning finish.

Usage::

    uv run python scripts/run_trajectory_curvature_v0_6.py --probe      # timing probe
    uv run python scripts/run_trajectory_curvature_v0_6.py              # full protocol
    uv run python scripts/run_trajectory_curvature_v0_6.py --reuse-checkpoints
"""

from __future__ import annotations

import os

# Small per-coordinate models: one XLA thread per worker process, many processes.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from neuralopt import Problem, minimize  # noqa: E402
from neuralopt.features import (  # noqa: E402
    denormalize_coordinates,
    gradient_to_normalized,
    normalize_coordinates,
)
from neuralopt.model import ModelConfig  # noqa: E402
from neuralopt.tasks import analytic_task, quadratic_task  # noqa: E402
from neuralopt.training import (  # noqa: E402
    HistoryUnrollConfig,
    load_checkpoint,
    save_checkpoint,
    train,
)
from neuralopt.trajectory import (  # noqa: E402
    empty_history,
    preconditioned_candidate,
    push_history,
    secant_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.6.0"
GENERATOR = "scaled_task/v0.6-1"
RESULTS = ROOT / "results" / f"trajectory-curvature-v{VERSION}.json"
FROZEN_V05 = ROOT / "artifacts" / "neuralopt-v0.5.0-overdamped_full.npz"

# ---- frozen protocol -------------------------------------------------------------
TRAIN_VARIANTS = (
    ("isotropic", 1.0),
    ("diagonal", 10.0),
    ("diagonal", 100.0),
    ("diagonal", 1000.0),
    ("rotated", 10.0),
    ("rotated", 100.0),
    ("rastrigin", 0.0),
)
TRAIN_DIMS, VALIDATION_DIMS, TEST_DIMS = (2, 4, 8), (3, 6), (7, 10, 16)
TEST_FAMILIES = (
    ("isotropic", 1.0, "in_distribution"),
    ("diagonal", 1000.0, "in_distribution"),
    ("rotated", 100.0, "in_distribution"),
    ("rastrigin", 0.0, "in_distribution"),
    ("rotated", 1000.0, "held_out_family"),
    ("rosenbrock", 0.0, "held_out_family"),
    ("ackley", 0.0, "held_out_family"),
)
TEST_REPLICATES = 4
CURVATURE_DOMAIN = ("diagonal-1000", "rotated-100", "rotated-1000")
BUDGETS, PRIMARY_BUDGET = (50, 100, 200), 100
TRAINING_SEEDS = (0, 1, 2)
UPDATES = 400
FIXED_STEPS = (0.01, 0.03, 0.1, 0.3, 1.0)
REGRET_FLOOR = 1e-10
BOOTSTRAP_DRAWS = 10_000
PRIMARY = "tf-h8"
TF_BASE = dict(
    width=16, heads=4, rank=2, layers=1, scale_aware=True, geometry_bound=10.0,
    lowrank_scale=0.5,
)
MLP_BASE = {**TF_BASE, "width": 48}
TRAINED_ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "tf-legacy": ("transformer", TF_BASE),
    "tf-h0": ("transformer", {**TF_BASE, "feature_set": "history-v1", "history_length": 0}),
    "tf-h4": ("transformer", {**TF_BASE, "feature_set": "history-v1", "history_length": 4}),
    "tf-h8": ("transformer", {**TF_BASE, "feature_set": "history-v1", "history_length": 8}),
    "mlp-h0": ("mlp", {**MLP_BASE, "feature_set": "history-v1", "history_length": 0}),
    "mlp-h4": ("mlp", {**MLP_BASE, "feature_set": "history-v1", "history_length": 4}),
    "mlp-h8": ("mlp", {**MLP_BASE, "feature_set": "history-v1", "history_length": 8}),
}
REFERENCE_ARMS = ("fixed", "bb", "lbfgs", "v05-frozen")


# ---- tasks ---------------------------------------------------------------------
def family_name(family: str, condition: float) -> str:
    return f"{family}-{int(condition)}" if family in ("diagonal", "rotated") else family


def task_specs(split: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if split in ("train", "validation"):
        dims, base = (TRAIN_DIMS, 1000) if split == "train" else (VALIDATION_DIMS, 2000)
        for i, (family, condition) in enumerate(TRAIN_VARIANTS):
            for j, dim in enumerate(dims):
                specs.append(_spec(split, family, condition, dim, base + 10 * i + j, i + j, ""))
    else:
        for fi, (family, condition, kind) in enumerate(TEST_FAMILIES):
            for di, dim in enumerate(TEST_DIMS):
                for r in range(TEST_REPLICATES):
                    seed = 3000 + 100 * fi + 10 * di + r
                    specs.append(_spec(split, family, condition, dim, seed, fi + di + r, kind))
    return specs


def _spec(
    split: str, family: str, condition: float, dim: int, seed: int, index: int, kind: str
) -> dict[str, Any]:
    return {
        "split": split,
        "family": family_name(family, condition),
        "base_family": family,
        "condition": condition,
        "dimension": dim,
        "seed": seed,
        "objective_scale": (0.2, 1.0, 5.0)[index % 3],
        "coordinate_scale": (0.5, 1.0, 2.0)[(2 * index + 1) % 3],
        "kind": kind,
    }


def split_hash(specs: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()


def build_task(spec: dict[str, Any]) -> Problem:
    family, dim, seed = spec["base_family"], spec["dimension"], spec["seed"]
    if family in ("isotropic", "diagonal", "rotated"):
        base = quadratic_task(
            jax.random.PRNGKey(seed), dim, spec["condition"], rotated=family == "rotated"
        )
    else:
        base = analytic_task(family, dim, seed)
    lo, hi = base.bounds
    a, c = spec["objective_scale"], spec["coordinate_scale"]

    def objective(x: Any) -> Any:
        return a * base.objective(x / c)

    assert base.x0 is not None
    return Problem(
        objective,
        (lo * c, hi * c),
        dim,
        x0=base.x0 * c,
        identifier=spec["family"],
        metadata={"known_optimum": a * float(base.metadata["known_optimum"]), **spec},
    )


# ---- references (harness only; never part of NeuralOpt inference) ----------------
def reference_loop(problem: Problem, budget: int, step_rule: str, fixed_step: float) -> dict:
    """Same acceptance/shrink/accounting rule as ``candidate_method='preconditioned'``.

    ``step_rule='fixed'`` reproduces the package's fixed loop (self-checked);
    ``'bb'`` uses the Barzilai--Borwein length ``‖g_u‖ / R_last`` (capped at 1) when
    the newest valid secant has ``R = sᵀy/sᵀs > 0``, otherwise ``fixed_step``.
    """
    lo, hi = problem.bounds
    objective = jax.jit(problem.objective)
    gradient = jax.jit(jax.grad(problem.objective))
    x = problem.x0
    value, grad = float(objective(x)), gradient(x)
    used = 2
    hist = push_history(
        empty_history(1, problem.dimension), normalize_coordinates(x, lo, hi),
        gradient_to_normalized(grad, lo, hi), value,
    )
    shrink, trajectory, accepted_steps = 1.0, [value], 0
    objective_count, gradient_count = 1, 1
    while used + 2 <= budget:
        g_u = gradient_to_normalized(grad, lo, hi)
        norm = float(jnp.linalg.norm(g_u))
        if norm == 0.0:
            break
        step = fixed_step
        if step_rule == "bb":
            s, y, mask = secant_pairs(hist)
            if bool(mask[-1]):
                rayleigh = float(s[-1] @ y[-1] / (s[-1] @ s[-1]))
                if rayleigh > 0:
                    step = min(norm / rayleigh, 1.0)
        u_new = preconditioned_candidate(
            normalize_coordinates(x, lo, hi), g_u / norm, jnp.ones_like(g_u),
            jnp.zeros((problem.dimension, 1), g_u.dtype), step, shrink,
        )
        candidate = jnp.clip(denormalize_coordinates(u_new, lo, hi), lo, hi)
        candidate_value = float(objective(candidate))
        used += 1
        objective_count += 1
        accepted = bool(np.isfinite(candidate_value) and candidate_value < value)
        if accepted:
            candidate_grad = gradient(candidate)
            used += 1
            gradient_count += 1
            accepted = bool(jnp.all(jnp.isfinite(candidate_grad)))
        if accepted:
            x, value, grad, shrink = candidate, candidate_value, candidate_grad, 1.0
            accepted_steps += 1
            hist = push_history(
                hist, normalize_coordinates(x, lo, hi), gradient_to_normalized(grad, lo, hi),
                value,
            )
        else:
            shrink *= 0.5
        trajectory.append(value)
    return {
        "fun": value,
        "trajectory": trajectory,
        "objective_evaluations": objective_count,
        "gradient_evaluations": gradient_count,
        "accepted_steps": accepted_steps,
    }


class _Exhausted(Exception):
    pass


def lbfgs_reference(problem: Problem, budget: int) -> dict:
    """SciPy L-BFGS-B; every call costs one objective plus one gradient unit."""
    from scipy.optimize import minimize as scipy_minimize

    value_and_grad = jax.jit(jax.value_and_grad(problem.objective))
    lo, hi = (np.asarray(b, dtype=float) for b in problem.bounds)
    calls, best, trajectory = 0, np.inf, []

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls, best
        if calls >= budget // 2:
            raise _Exhausted
        calls += 1
        v, g = value_and_grad(jnp.asarray(x, jnp.float32))
        v = float(v)
        if np.isfinite(v):
            best = min(best, v)
        trajectory.append(best)
        return v, np.asarray(g, dtype=float)

    try:
        scipy_minimize(
            fun, np.asarray(problem.x0, dtype=float), jac=True, method="L-BFGS-B",
            bounds=list(zip(lo, hi, strict=True)),
            options={"maxfun": budget // 2, "maxiter": budget, "ftol": 0.0, "gtol": 0.0},
        )
    except _Exhausted:
        pass
    return {
        "fun": float(best),
        "trajectory": trajectory,
        "objective_evaluations": calls,
        "gradient_evaluations": calls,
        "accepted_steps": None,
    }


# ---- metrics ---------------------------------------------------------------------
def log_regret(value: float, f0: float, optimum: float) -> float:
    regret = max(value - optimum, 0.0) / (abs(f0 - optimum) + 1e-12)
    return float(np.log10(regret + REGRET_FLOOR)) if np.isfinite(value) else float("nan")


def units_to_target(trajectory: list[float], f0: float, optimum: float, lbfgs: bool) -> Any:
    """Objective+gradient units spent when normalized regret first reaches 1e-3."""
    target = optimum + 1e-3 * (abs(f0 - optimum) + 1e-12)
    used, previous = 2, trajectory[0] if not lbfgs else np.inf
    for i, value in enumerate(trajectory if lbfgs else trajectory[1:]):
        if lbfgs:
            used = 2 * (i + 1)
        else:
            used += 1 + int(value < previous)
            previous = value
        if value <= target:
            return used
    return None


def evaluate_once(
    arm: str, problem: Problem, budget: int, job: dict[str, Any], params: Any, config: Any,
    model_kind: str,
) -> dict[str, Any]:
    if arm == "lbfgs":
        return lbfgs_reference(problem, budget)
    if arm == "bb":
        return reference_loop(problem, budget, "bb", job["fixed_step"])
    fixed = arm == "fixed"
    result = minimize(
        problem,
        max_evaluations=budget,
        params=params,
        config=config,
        model_kind=model_kind,  # type: ignore[arg-type]
        candidate_method="preconditioned",
        hamiltonian_metric="fixed" if fixed else "learned",
        fixed_metric_diag=np.ones(problem.dimension) if fixed else None,
        preconditioned_step=job["fixed_step"] if fixed else None,
    )
    d = result.diagnostics or {}
    return {
        "fun": result.fun,
        "trajectory": list(result.trajectory),
        "objective_evaluations": d["objective_evaluations"],
        "gradient_evaluations": d["gradient_evaluations"],
        "accepted_steps": d["accepted_steps"],
    }


def run_arm(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate one arm (and one training seed) on every final-test task and budget."""
    arm, seed = job["arm"], job.get("seed")
    params = config = None
    model_kind = "transformer"
    if arm in TRAINED_ARMS:
        params, config = load_checkpoint(job["checkpoint"])
        model_kind = TRAINED_ARMS[arm][0]
    elif arm == "v05-frozen":
        params, config = load_checkpoint(FROZEN_V05)
    rows = []
    for spec in task_specs("test"):
        problem = build_task(spec)
        optimum = float(problem.metadata["known_optimum"])
        f0 = float(problem.objective(problem.x0))
        for budget in BUDGETS:
            repeats = 2 if budget == PRIMARY_BUDGET else 1
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                try:
                    out = evaluate_once(arm, problem, budget, job, params, config, model_kind)
                except (FloatingPointError, ValueError) as exc:  # counted, never dropped
                    out = {"fun": f0, "trajectory": [f0], "objective_evaluations": None,
                           "gradient_evaluations": None, "accepted_steps": None,
                           "error": f"{type(exc).__name__}: {exc}"}
                timings.append(time.perf_counter() - start)
            rows.append(
                {
                    "arm": arm,
                    "training_seed": seed,
                    "task_seed": spec["seed"],
                    "family": spec["family"],
                    "kind": spec["kind"],
                    "dimension": spec["dimension"],
                    "objective_scale": spec["objective_scale"],
                    "coordinate_scale": spec["coordinate_scale"],
                    "budget": budget,
                    "initial_objective": f0,
                    "known_optimum": optimum,
                    "best_objective": float(out["fun"]),
                    "log10_regret": log_regret(float(out["fun"]), f0, optimum),
                    "finite": bool(np.isfinite(out["fun"])) and "error" not in out,
                    "error": out.get("error"),
                    "units_to_1e-3": units_to_target(
                        out["trajectory"], f0, optimum, arm == "lbfgs"
                    ),
                    "objective_evaluations": out["objective_evaluations"],
                    "gradient_evaluations": out["gradient_evaluations"],
                    "hvp_evaluations": 0,
                    "accepted_steps": out["accepted_steps"],
                    "first_seconds": timings[0],
                    "warm_seconds": timings[-1] if repeats > 1 else None,
                }
            )
    return rows


# ---- training --------------------------------------------------------------------
def checkpoint_path(arm: str, seed: int) -> Path:
    return ROOT / "artifacts" / f"trajectory-v{VERSION}-{arm}-seed{seed}.npz"


def train_arm(job: dict[str, Any]) -> dict[str, Any]:
    arm, seed, updates = job["arm"], job["seed"], job["updates"]
    model_kind, kwargs = TRAINED_ARMS[arm]
    config = ModelConfig(**kwargs)
    specs = task_specs("train")
    problems = [build_task(s) for s in specs]
    start = time.perf_counter()
    params, losses = train(
        problems,
        seed=seed,
        steps=updates,
        config=config,
        objective="history",
        history_unroll=HistoryUnrollConfig(),
        model_kind=model_kind,
    )
    seconds = time.perf_counter() - start
    info = {
        "arm": arm,
        "seed": seed,
        "model_kind": model_kind,
        "updates": updates,
        "seconds": seconds,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_50_loss": float(np.mean(losses[-50:])),
        "parameter_count": int(sum(np.asarray(v).size for v in params.values())),
        "train_split_sha256": split_hash(specs),
        "validation_split_sha256": split_hash(task_specs("validation")),
        "generator": GENERATOR,
        "rng": f"jax.random.PRNGKey({seed}) in neuralopt.training.train",
        "git_head": job["git_head"],
    }
    if job.get("save", True):
        path = checkpoint_path(arm, seed)
        save_checkpoint(path, params, config, info)
        info["checkpoint_bytes"] = path.stat().st_size
    return info


# ---- statistics ------------------------------------------------------------------
def hierarchical_bootstrap(matrix: np.ndarray, seed: int = 0) -> list[float]:
    """95% CI of the mean of ``matrix[training_seed, task]``, resampling both levels."""
    if not np.all(np.isfinite(matrix)):
        raise ValueError("nonfinite paired differences; failures must be resolved explicitly")
    rng = np.random.default_rng(seed)
    n_seeds, n_tasks = matrix.shape
    seeds = rng.integers(0, n_seeds, (BOOTSTRAP_DRAWS, n_seeds))
    tasks = rng.integers(0, n_tasks, (BOOTSTRAP_DRAWS, n_tasks))
    draws = np.array([matrix[s][:, t].mean() for s, t in zip(seeds, tasks, strict=True)])
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def difference_matrix(
    rows: list[dict], candidate: str, reference: str, budget: int, families: tuple[str, ...]
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    def table(arm: str) -> dict[Any, dict[Any, float]]:
        out: dict[Any, dict[Any, float]] = {}
        for r in rows:
            if r["arm"] == arm and r["budget"] == budget and r["family"] in families:
                key = (r["family"], r["dimension"], r["task_seed"])
                out.setdefault(r["training_seed"], {})[key] = r["log10_regret"]
        return out

    cand, ref = table(candidate), table(reference)
    tasks = sorted(next(iter(cand.values())).keys())
    cand_seeds = sorted(cand, key=lambda s: -1 if s is None else s)
    ref_seeds = sorted(ref, key=lambda s: -1 if s is None else s)
    matrix = []
    for i, s in enumerate(cand_seeds):
        r_seed = ref_seeds[i] if len(ref_seeds) == len(cand_seeds) else ref_seeds[0]
        matrix.append([cand[s][t] - ref[r_seed][t] for t in tasks])
    return np.asarray(matrix, dtype=float), tasks


def compare(
    rows: list[dict], candidate: str, reference: str, budget: int, families: tuple[str, ...]
) -> dict[str, Any]:
    matrix, tasks = difference_matrix(rows, candidate, reference, budget, families)
    return {
        "candidate": candidate,
        "reference": reference,
        "budget": budget,
        "families": list(families),
        "tasks": len(tasks),
        "training_seeds": int(matrix.shape[0]),
        "mean_difference_log10": float(np.mean(matrix)),
        "ci95": hierarchical_bootstrap(matrix),
        "wins": int(np.sum(matrix < 0)),
        "pairs": int(matrix.size),
    }


def summarize(rows: list[dict]) -> dict[str, Any]:
    families = tuple(family_name(f, c) for f, c, _ in TEST_FAMILIES)
    held_out = tuple(family_name(f, c) for f, c, k in TEST_FAMILIES if k == "held_out_family")
    arms = list(TRAINED_ARMS) + ["bb", "lbfgs", "v05-frozen"]
    primary = {
        "H1_history_value": compare(rows, PRIMARY, "tf-h0", PRIMARY_BUDGET, CURVATURE_DOMAIN),
        "H2_attention_value": compare(rows, PRIMARY, "mlp-h8", PRIMARY_BUDGET, CURVATURE_DOMAIN),
        "H3_vs_fixed": compare(rows, PRIMARY, "fixed", PRIMARY_BUDGET, CURVATURE_DOMAIN),
    }
    family_regressions = {
        fam: compare(rows, PRIMARY, "fixed", PRIMARY_BUDGET, (fam,)) for fam in families
    }
    major = [
        fam for fam, c in family_regressions.items()
        if c["mean_difference_log10"] > 0.5 and c["ci95"][0] > 0
    ]
    outcomes = {
        "H1": primary["H1_history_value"]["ci95"][1] < 0,
        "H2": primary["H2_attention_value"]["ci95"][1] < 0,
        "H3": primary["H3_vs_fixed"]["ci95"][1] < 0 and not major,
        "H3_major_family_regressions": major,
    }
    secondary = []
    for budget in BUDGETS:
        for domain_name, domain in (
            ("curvature", CURVATURE_DOMAIN), ("all", families), ("held_out_families", held_out)
        ):
            for arm in arms:
                for ref in ("fixed", "lbfgs", "bb"):
                    if arm != ref:
                        secondary.append({"domain": domain_name, **compare(
                            rows, arm, ref, budget, domain)})
    for cand, ref in (("tf-h4", "tf-h0"), ("mlp-h8", "mlp-h0"), ("mlp-h4", "mlp-h0"),
                      ("tf-h8", "tf-h4"), ("tf-h0", "tf-legacy"), ("tf-h4", "mlp-h4")):
        secondary.append({"domain": "curvature", **compare(
            rows, cand, ref, PRIMARY_BUDGET, CURVATURE_DOMAIN)})
    per_family = {
        arm: {
            fam: float(np.mean([r["log10_regret"] for r in rows if r["arm"] == arm
                                and r["family"] == fam and r["budget"] == PRIMARY_BUDGET]))
            for fam in families
        }
        for arm in ["fixed", *arms]
    }
    costs = {}
    for arm in ["fixed", *arms]:
        sel = [r for r in rows if r["arm"] == arm and r["budget"] == PRIMARY_BUDGET]
        warm = [r["warm_seconds"] for r in sel if r["warm_seconds"] is not None]
        first = [r["first_seconds"] for r in rows if r["arm"] == arm and r["budget"] == BUDGETS[0]]
        costs[arm] = {
            "mean_objective_evaluations": float(np.mean([r["objective_evaluations"] for r in sel])),
            "mean_gradient_evaluations": float(np.mean([r["gradient_evaluations"] for r in sel])),
            "hvp_evaluations": 0,
            "max_units": int(max(r["objective_evaluations"] + r["gradient_evaluations"]
                                 for r in sel)),
            "nonfinite_or_failed_rate": float(np.mean([not r["finite"] for r in sel])),
            "failures": sum(r["error"] is not None for r in rows if r["arm"] == arm),
            "success_rate_1e-3": float(np.mean([r["units_to_1e-3"] is not None for r in sel])),
            # First call on each task (budget 50 runs first): includes objective
            # compilation; model compilation is shared after the first task per process.
            "median_first_call_seconds": float(np.median(first)),
            "max_first_call_seconds": float(np.max(first)),
            "median_warm_seconds": float(np.median(warm)) if warm else None,
        }
    return {
        "primary": primary,
        "outcomes": outcomes,
        "family_vs_fixed": family_regressions,
        "per_family_mean_log10_regret": per_family,
        "costs": costs,
        "secondary": secondary,
    }


# ---- orchestration ---------------------------------------------------------------
def tune_fixed_step() -> dict[str, Any]:
    specs = task_specs("validation")
    scores = {}
    for step in FIXED_STEPS:
        values = []
        for spec in specs:
            problem = build_task(spec)
            f0 = float(problem.objective(problem.x0))
            out = reference_loop(problem, PRIMARY_BUDGET, "fixed", step)
            values.append(log_regret(out["fun"], f0, float(problem.metadata["known_optimum"])))
        scores[str(step)] = float(np.mean(values))
    selected = min(FIXED_STEPS, key=lambda s: scores[str(s)])
    return {"candidates": list(FIXED_STEPS), "validation_mean_log10_regret": scores,
            "selected": selected, "validation_split_sha256": split_hash(specs)}


def self_check_reference_parity() -> None:
    """The harness fixed loop must equal the package's fixed preconditioned loop."""
    problem = build_task(task_specs("validation")[5])
    ours = reference_loop(problem, 40, "fixed", 0.1)
    package = minimize(
        problem, max_evaluations=40, candidate_method="preconditioned",
        hamiltonian_metric="fixed", fixed_metric_diag=np.ones(problem.dimension),
        preconditioned_step=0.1,
    )
    np.testing.assert_allclose(ours["trajectory"], package.trajectory, rtol=1e-6)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict[str, Any]:
    sources = sorted((ROOT / "src" / "neuralopt").glob("*.py")) + [Path(__file__).resolve()]
    return {
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
        "design_sha256": sha256_file(
            ROOT / "docs/superpowers/specs/2026-11-21-trajectory-curvature-v06-design.md"
        ),
        "checkpoint_sha256": {
            p.name: sha256_file(p)
            for p in sorted((ROOT / "artifacts").glob(f"trajectory-v{VERSION}-*.npz"))
        },
    }


def git_head() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                           cwd=ROOT)
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))


def probe() -> None:
    job = {"arm": PRIMARY, "seed": 0, "updates": 20, "git_head": git_head(), "save": False}
    info = train_arm(job)
    print(json.dumps({"probe": info, "projected_seconds_for_updates": info["seconds"]
                      / 20 * UPDATES}, indent=2))


def main(reuse: bool, workers: int) -> None:
    head = git_head()
    self_check_reference_parity()
    jobs = [
        {"arm": arm, "seed": seed, "updates": UPDATES, "git_head": head}
        for arm in TRAINED_ARMS for seed in TRAINING_SEEDS
    ]
    training: list[dict[str, Any]] = []
    todo = [j for j in jobs if not (reuse and checkpoint_path(j["arm"], j["seed"]).exists())]
    start = time.perf_counter()
    with pool(workers) as executor:
        training = list(executor.map(train_arm, todo))
    training_wall = time.perf_counter() - start
    for job in jobs:  # reused checkpoints contribute their recorded provenance
        if job in todo:
            continue
        with np.load(checkpoint_path(job["arm"], job["seed"])) as data:
            info = ast.literal_eval(str(data["metadata"][0]))
        info["checkpoint_bytes"] = checkpoint_path(job["arm"], job["seed"]).stat().st_size
        training.append(info)
    fixed = tune_fixed_step()  # validation only, before any final-test task is built
    eval_jobs = [
        {"arm": arm, "seed": seed, "checkpoint": str(checkpoint_path(arm, seed))}
        for arm in TRAINED_ARMS for seed in TRAINING_SEEDS
    ] + [{"arm": arm, "fixed_step": fixed["selected"]} for arm in REFERENCE_ARMS]
    start = time.perf_counter()
    with pool(workers) as executor:
        rows = [row for chunk in executor.map(run_arm, eval_jobs) for row in chunk]
    evaluation_wall = time.perf_counter() - start
    test_specs = task_specs("test")
    output = {
        "schema": "neuralopt-trajectory-curvature-results/v1",
        "environment": {
            "jax": jax.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "git_head": head,
            "worker_processes": workers,
            **provenance(),
        },
        "protocol": {
            "design": "docs/superpowers/specs/2026-11-21-trajectory-curvature-v06-design.md",
            "generator": GENERATOR,
            "train_split_sha256": split_hash(task_specs("train")),
            "validation_split_sha256": split_hash(task_specs("validation")),
            "test_split_sha256": split_hash(test_specs),
            "test_tasks": len(test_specs),
            "curvature_domain": list(CURVATURE_DOMAIN),
            "budgets": list(BUDGETS),
            "primary_budget": PRIMARY_BUDGET,
            "budget_units": "objective evaluations + gradient evaluations",
            "training_seeds": list(TRAINING_SEEDS),
            "updates": UPDATES,
            "history_unroll": HistoryUnrollConfig().__dict__,
            "optimizer": "optax chain(clip_by_global_norm(1.0), adam(1e-3)); one task per update",
            "fixed_comparator": "scalar step L on P = I (normalized-gradient step), "
            "validation-tuned at budget 100 and reused at 50/200",
            "primary_candidate": PRIMARY,
            "metric": "log10(max(f-f*,0)/(|f0-f*|+1e-12) + 1e-10); difference cand-ref",
            "bootstrap": f"two-level (training seed, task), {BOOTSTRAP_DRAWS} draws",
        },
        "fixed_tuning": fixed,
        "training": training,
        "training_wall_seconds": training_wall if todo else None,
        "evaluation_wall_seconds": evaluation_wall,
        "summary": summarize(rows),
        "rows": rows,
    }
    RESULTS.write_text(json.dumps(output, indent=1))
    print(json.dumps({k: output["summary"][k] for k in ("primary", "outcomes")}, indent=2))


def _fmt(c: dict[str, Any]) -> str:
    lo, hi = c["ci95"]
    return f"{c['mean_difference_log10']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def tables(path: Path) -> None:
    """Print the report's markdown tables from a results file (no recomputation of runs)."""
    data = json.loads(path.read_text())
    s, rows = data["summary"], data["rows"]
    print("| Hypothesis | Contrast (curvature domain, budget 100) | Mean D (95% CI) | Wins | "
          "Pre-registered outcome |\n|---|---|---:|---:|---|")
    names = {"H1_history_value": "H1", "H2_attention_value": "H2", "H3_vs_fixed": "H3"}
    for key, c in s["primary"].items():
        print(f"| {names[key]} | {c['candidate']} − {c['reference']} | {_fmt(c)} | "
              f"{c['wins']}/{c['pairs']} | "
              f"{'supported' if s['outcomes'][names[key]] else 'not supported'} |")
    families = list(s["family_vs_fixed"])
    print("\n| Arm | " + " | ".join(families) + " |\n|---|" + "---:|" * len(families))
    for arm, values in s["per_family_mean_log10_regret"].items():
        print(f"| {arm} | " + " | ".join(f"{values[f]:.2f}" for f in families) + " |")
    print("\n| Family | tf-h8 − fixed (95% CI) |\n|---|---:|")
    for fam, c in s["family_vs_fixed"].items():
        print(f"| {fam} | {_fmt(c)} |")
    print("\n| Candidate | Reference | Budget 50 | Budget 100 | Budget 200 |"
          "\n|---|---|---:|---:|---:|")
    sec = {(c["domain"], c["candidate"], c["reference"], c["budget"]): c for c in s["secondary"]}
    for cand in [*TRAINED_ARMS, "bb", "v05-frozen"]:
        for ref in ("fixed", "bb", "lbfgs"):
            if (("curvature", cand, ref, 100)) in sec:
                cells = [_fmt(sec[("curvature", cand, ref, b)]) for b in BUDGETS]
                print(f"| {cand} | {ref} | " + " | ".join(cells) + " |")
    print("\n| Contrast (curvature, budget 100) | Mean D (95% CI) |\n|---|---:|")
    for c in s["secondary"]:
        if c["reference"] not in ("fixed", "bb", "lbfgs"):
            print(f"| {c['candidate']} − {c['reference']} | {_fmt(c)} |")
    print("\n| Arm | Obj | Grad | HVP | Fail | Success 1e-3 | First call s (median/max) | Warm s |"
          "\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm, c in s["costs"].items():
        print(f"| {arm} | {c['mean_objective_evaluations']:.1f} | "
              f"{c['mean_gradient_evaluations']:.1f} | 0 | {c['failures']} | "
              f"{c['success_rate_1e-3']:.2f} | {c['median_first_call_seconds']:.2f}/"
              f"{c['max_first_call_seconds']:.2f} | {c['median_warm_seconds']:.3f} |")
    # Post-hoc (NOT pre-registered): float32 objectives hit exact zeros, so the 1e-10
    # floor can dominate means. Recompute H3 and its family guard with floor 1e-6.
    def posthoc(row: dict[str, Any]) -> float:
        regret = max(row["best_objective"] - row["known_optimum"], 0.0) / (
            abs(row["initial_objective"] - row["known_optimum"]) + 1e-12)
        return float(np.log10(regret + 1e-6))

    adjusted = [{**r, "log10_regret": posthoc(r)} for r in rows]
    print("\n| POST-HOC floor 1e-6 | tf-h8 − fixed (95% CI) |\n|---|---:|")
    curvature = compare(adjusted, PRIMARY, "fixed", 100, CURVATURE_DOMAIN)
    print(f"| curvature domain | {_fmt(curvature)} |")
    for fam in families:
        print(f"| {fam} | {_fmt(compare(adjusted, PRIMARY, 'fixed', 100, (fam,)))} |")
    zeros = sum(r["best_objective"] == r["known_optimum"] for r in rows)
    print(f"\nRows with best objective exactly equal to the optimum (float32): {zeros}/{len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tables", action="store_true", help="print report tables from results")
    args = parser.parse_args()
    if args.tables:
        tables(RESULTS)
    elif args.probe:
        probe()
    else:
        main(args.reuse_checkpoints, args.workers)
