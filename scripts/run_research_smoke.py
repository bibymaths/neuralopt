"""Reproducible, CPU-sized v0.2 trajectory and meta-training smoke experiment."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import jax
import numpy as np

from neuralopt import Problem, minimize
from neuralopt.model import DEFAULT_CONFIG, init_model
from neuralopt.tasks import analytic_task, quadratic_task, task_split
from neuralopt.training import UnrollConfig, load_checkpoint, save_checkpoint, train


def controls(problem: Problem, budget: int) -> float:
    x = problem.x0
    lo, hi = problem.bounds
    grad = problem.gradient or jax.grad(problem.objective)
    best = float(problem.objective(x))
    for _ in range(budget - 1):
        g = grad(x)
        x = np.asarray(x - 0.05 * (hi - lo) * g / (np.linalg.norm(g) + 1e-8))
        x = np.clip(x, lo, hi)
        best = min(best, float(problem.objective(x)))
    return best


def summarize(result: object) -> dict[str, float]:
    records = result.diagnostics["records"]
    update = [r for r in records if r["kind"] == "update"]
    proposal = [r for r in records if r["kind"] == "proposal"]
    return {
        "final_objective": result.fun,
        "acceptance_rate": result.diagnostics["acceptance_rate"],
        "update_raw_alignment": float(np.median([r["raw_alignment"] for r in update])),
        "update_preconditioned_alignment": float(
            np.median([r["preconditioned_alignment"] for r in update])
        ),
        "proposal_acceptance": float(np.mean([r["accepted"] for r in proposal])),
        "projection_fraction": float(np.mean([r["projected_fraction"] for r in records])),
        "preconditioner_condition": float(
            np.median([r["preconditioner_condition"] for r in records])
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    old, config = load_checkpoint(root / "artifacts/demo-v0.1.0.npz")
    random = init_model(jax.random.PRNGKey(123), DEFAULT_CONFIG)
    tasks = [
        ("isotropic", quadratic_task(jax.random.PRNGKey(10), 4, 1.0)),
        ("ill_conditioned", quadratic_task(jax.random.PRNGKey(11), 4, 1e3)),
        ("rotated", quadratic_task(jax.random.PRNGKey(12), 4, 1e2, True)),
        ("rosenbrock", analytic_task("rosenbrock", 4)),
        ("rastrigin", analytic_task("rastrigin", 4)),
        ("ackley", analytic_task("ackley", 4)),
    ]
    rows = []
    for family, problem in tasks:
        control = controls(problem, 50)
        for algorithm, params in (("v0.1.0", old), ("random", random)):
            start = time.perf_counter()
            result = minimize(
                problem,
                max_evaluations=50,
                params=params,
                config=config,
                record_diagnostics=True,
            )
            rows.append(
                {
                    "family": family,
                    "algorithm": algorithm,
                    "dimension": problem.dimension,
                    "budget": 50,
                    "runtime_seconds": time.perf_counter() - start,
                    "gradient_descent_control": control,
                    **summarize(result),
                }
            )

    train_tasks = [
        r.problem
        for dimension in (2, 4)
        for r in task_split(100 + dimension, 6, dimension)
        if r.split == "train"
    ]
    start = time.perf_counter()
    learned, losses = train(
        train_tasks,
        seed=7,
        steps=12,
        learning_rate=3e-4,
        objective="regret",
        unroll=UnrollConfig(steps=10, horizons=(2, 5, 10)),
    )
    training_seconds = time.perf_counter() - start
    checkpoint = root / "artifacts/neuralopt-v0.2.0-smoke.npz"
    save_checkpoint(
        checkpoint,
        learned,
        config,
        {"objective": "regret", "updates": 12, "unroll": 10, "seed": 7},
    )
    heldout = quadratic_task(jax.random.PRNGKey(999), 8, 100.0, True)
    trained_result = minimize(heldout, max_evaluations=50, params=learned, config=config)
    old_result = minimize(heldout, max_evaluations=50, params=old, config=config)
    payload = {
        "schema": "neuralopt.research-smoke.v1",
        "environment": {"platform": platform.platform(), "jax": jax.__version__},
        "diagnostics": rows,
        "training": {
            "updates": 12,
            "unroll_steps": 10,
            "seconds": training_seconds,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "heldout_dimension": 8,
            "heldout_rotated_condition": 100.0,
            "heldout_v0.1_objective": old_result.fun,
            "heldout_v0.2_smoke_objective": trained_result.fun,
        },
    }
    output = root / "results/research-smoke-v0.2.0.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["training"], indent=2))


if __name__ == "__main__":
    main()
