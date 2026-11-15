"""Machine-readable analytic benchmark runner."""

from __future__ import annotations

import json
import platform
import time
from typing import Any

import jax
import jax.numpy as jnp
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
    if budget < 1:
        raise ValueError("budget must be positive")
    lo, hi = (np.asarray(v, dtype=float) for v in problem.bounds)
    x0 = np.asarray(problem.x0, dtype=float)
    objective_calls = 0
    gradient_calls = 0
    best_x: np.ndarray | None = None
    best_value = np.inf

    class ObjectiveBudgetExhausted(RuntimeError):
        pass

    def objective(x: Any) -> float:
        nonlocal best_value, best_x, objective_calls
        if objective_calls >= budget:
            raise ObjectiveBudgetExhausted
        objective_calls += 1
        point = np.asarray(x, dtype=float)
        value = float(problem.objective(jnp.asarray(point, dtype=problem.dtype)))
        if np.isfinite(value) and value < best_value:
            best_x, best_value = point.copy(), value
        return value

    grad_fn = problem.gradient or jax.grad(problem.objective)

    def gradient(x: Any) -> np.ndarray:
        nonlocal gradient_calls
        gradient_calls += 1
        point = jnp.asarray(x, dtype=problem.dtype)
        return np.asarray(grad_fn(point), dtype=float)

    start = time.perf_counter()
    if algorithm == "random":
        rng = np.random.default_rng(seed)
        candidates = np.vstack((x0, rng.uniform(lo, hi, size=(budget - 1, problem.dimension))))
        for candidate in candidates:
            objective(candidate)
        if best_x is None:
            raise FloatingPointError("objective produced no finite in-budget value")
        x, value, status = best_x, best_value, "budget_exhausted"
    elif algorithm == "adam":
        x, m, v = x0.copy(), np.zeros_like(x0), np.zeros_like(x0)
        objective(x)
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
            objective(x)
        assert best_x is not None
        x, value, status = best_x, best_value, "budget_exhausted"
    elif algorithm == "scipy-lbfgsb":
        try:
            from scipy.optimize import minimize as scipy_minimize
        except ImportError as exc:
            raise ImportError("install neuralopt[benchmark] for SciPy baselines") from exc
        try:
            result = scipy_minimize(
                objective,
                x0,
                jac=gradient,
                bounds=list(zip(lo, hi, strict=True)),
                method="L-BFGS-B",
                options={"maxfun": budget, "maxiter": budget},
            )
            status = "success" if result.success else "stopped"
        except ObjectiveBudgetExhausted:
            status = "budget_exhausted"
        if best_x is None:
            raise FloatingPointError("objective produced no finite in-budget value")
        x, value = best_x, best_value
    else:
        raise ValueError("algorithm must be random, adam, or scipy-lbfgsb")
    known_optimum = problem.metadata.get("known_optimum")
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
        "known_optimum": known_optimum,
        "regret": value - known_optimum if known_optimum is not None else None,
        "status": status,
        "hardware": platform.machine(),
        "precision": str(problem.dtype),
    }
