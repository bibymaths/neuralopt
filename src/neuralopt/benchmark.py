"""Machine-readable analytic benchmark runner."""

from __future__ import annotations

import json
import platform
import time
from typing import Any

import jax
import numpy as np

from .optimizer import minimize
from .problem import Problem


def run(
    problem: Problem, seeds: list[int], budget: int, params: Any = None
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        start = time.perf_counter()
        result = minimize(problem, max_evaluations=budget, seed=seed, params=params)
        rows.append(
            {
                "objective": problem.identifier or "anonymous",
                "seed": seed,
                "dimension": problem.dimension,
                "algorithm": "neuralopt",
                "evaluation_budget": budget,
                "final_objective": result.fun,
                "evaluations": result.evaluations,
                "runtime_seconds": time.perf_counter() - start,
                "status": result.status,
                "hardware": platform.machine(),
                "training_status": "provided_checkpoint" if params else "bundled_checkpoint",
                "certification_status": None,
            }
        )
    return rows


def write_json(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def run_baseline(problem: Problem, algorithm: str, budget: int, seed: int = 0) -> dict[str, Any]:
    """Run an explicitly counted benchmark baseline outside NeuralOpt inference."""
    lo, hi = (np.asarray(v, dtype=float) for v in problem.bounds)
    x0 = np.asarray(problem.x0, dtype=float)
    objective_calls = 0
    gradient_calls = 0

    def objective(x: Any) -> float:
        nonlocal objective_calls
        objective_calls += 1
        return float(problem.objective(x))

    grad_fn = problem.gradient or jax.grad(problem.objective)

    def gradient(x: Any) -> np.ndarray:
        nonlocal gradient_calls
        gradient_calls += 1
        return np.asarray(grad_fn(x), dtype=float)

    start = time.perf_counter()
    if algorithm == "random":
        rng = np.random.default_rng(seed)
        candidates = np.vstack((x0, rng.uniform(lo, hi, size=(budget - 1, problem.dimension))))
        values = np.asarray([objective(x) for x in candidates])
        index = int(np.argmin(values))
        x, value, status = candidates[index], float(values[index]), "budget_exhausted"
    elif algorithm == "adam":
        x, m, v = x0.copy(), np.zeros_like(x0), np.zeros_like(x0)
        value = objective(x)
        best_x, best = x.copy(), value
        for step in range(1, budget):
            g = gradient(x)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            x = np.clip(
                x
                - 0.03
                * (hi - lo)
                * (m / (1 - 0.9**step))
                / (np.sqrt(v / (1 - 0.999**step)) + 1e-8),
                lo,
                hi,
            )
            value = objective(x)
            if value < best:
                best, best_x = value, x.copy()
        x, value, status = best_x, best, "budget_exhausted"
    elif algorithm == "scipy-lbfgsb":
        try:
            from scipy.optimize import minimize as scipy_minimize
        except ImportError as exc:
            raise ImportError("install neuralopt[benchmark] for SciPy baselines") from exc
        result = scipy_minimize(
            objective,
            x0,
            jac=gradient,
            bounds=list(zip(lo, hi, strict=True)),
            method="L-BFGS-B",
            options={"maxfun": budget, "maxiter": budget},
        )
        x, value, status = result.x, float(result.fun), "success" if result.success else "stopped"
    else:
        raise ValueError("algorithm must be random, adam, or scipy-lbfgsb")
    return {
        "task_family": problem.identifier or "anonymous",
        "instance": problem.metadata.get("instance"),
        "dimension": problem.dimension,
        "algorithm": algorithm,
        "seed": seed,
        "evaluation_budget": budget,
        "objective_evaluations": objective_calls,
        "gradient_evaluations": gradient_calls,
        "hessian_vector_products": 0,
        "runtime_seconds": time.perf_counter() - start,
        "final_solution": np.asarray(x).tolist(),
        "best_objective": value,
        "known_optimum": problem.metadata.get("known_optimum"),
        "regret": value - problem.metadata.get("known_optimum", value),
        "status": status,
        "hardware": platform.machine(),
        "precision": str(problem.dtype),
    }
