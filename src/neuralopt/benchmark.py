"""Machine-readable analytic benchmark runner."""

from __future__ import annotations

import json
import platform
import time
from typing import Any

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
