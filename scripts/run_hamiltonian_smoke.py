"""Matched-objective-budget Hamiltonian candidate smoke experiment."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import jax
import numpy as np

from neuralopt import minimize
from neuralopt.hamiltonian import HamiltonianConfig
from neuralopt.tasks import analytic_task, quadratic_task
from neuralopt.training import load_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    params, model_config = load_checkpoint(root / "artifacts/demo-v0.1.0.npz")
    hmc_config = HamiltonianConfig(step_size=0.03, leapfrog_steps=2, inverse_temperature=1.0)
    rows = []
    for seed in (0, 1, 2):
        tasks = [
            ("isotropic", quadratic_task(jax.random.PRNGKey(10 + seed), 4, 1.0)),
            ("ill_conditioned", quadratic_task(jax.random.PRNGKey(20 + seed), 4, 1e3)),
            ("rotated", quadratic_task(jax.random.PRNGKey(30 + seed), 4, 1e2, True)),
            ("rastrigin_2d", analytic_task("rastrigin", 2, seed)),
            ("ackley_8d", analytic_task("ackley", 8, seed)),
        ]
        for family, problem in tasks:
            variants = (
                ("direct", "direct", "learned"),
                ("hmc_identity", "hamiltonian", "identity"),
                ("hmc_learned_metric", "hamiltonian", "learned"),
            )
            for algorithm, method, metric in variants:
                # First call includes compilation for this objective/shape; the
                # second call measures warm execution with identical randomness.
                start = time.perf_counter()
                minimize(
                    problem,
                    max_evaluations=100,
                    seed=seed,
                    params=params,
                    config=model_config,
                    candidate_method=method,
                    hamiltonian_metric=metric,
                    hamiltonian_config=hmc_config,
                    record_diagnostics=True,
                )
                first_call_seconds = time.perf_counter() - start
                start = time.perf_counter()
                result = minimize(
                    problem,
                    max_evaluations=100,
                    seed=seed,
                    params=params,
                    config=model_config,
                    candidate_method=method,
                    hamiltonian_metric=metric,
                    hamiltonian_config=hmc_config,
                    record_diagnostics=True,
                )
                diagnostics = result.diagnostics or {}
                records = diagnostics.get("records") or []
                energy = [abs(r["energy_error"]) for r in records if r["energy_error"] is not None]
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "dimension": problem.dimension,
                        "algorithm": algorithm,
                        "best_objective": result.fun,
                        "first_call_seconds": first_call_seconds,
                        "warm_runtime_seconds": time.perf_counter() - start,
                        "objective_evaluations": diagnostics["objective_evaluations"],
                        "gradient_evaluations": diagnostics["gradient_evaluations"],
                        "candidate_proposals": diagnostics["candidate_proposals"],
                        "transition_acceptance": diagnostics["acceptance_rate"],
                        "hmc_transition_acceptance": diagnostics["hmc_transition_acceptance"],
                        "objective_improvement_rate": diagnostics["objective_improvement_rate"],
                        "incumbent_update_rate": diagnostics["incumbent_update_rate"],
                        "duplicate_candidate_rate": diagnostics["duplicate_candidate_rate"],
                        "leapfrog_steps": diagnostics["leapfrog_steps"],
                        "median_absolute_energy_error": (
                            float(np.median(energy)) if energy else None
                        ),
                    }
                )
    summary = {}
    for algorithm in sorted({row["algorithm"] for row in rows}):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        summary[algorithm] = {
            "median_objective": float(np.median([row["best_objective"] for row in selected])),
            "median_runtime_seconds": float(
                np.median([row["warm_runtime_seconds"] for row in selected])
            ),
            "median_first_call_seconds": float(
                np.median([row["first_call_seconds"] for row in selected])
            ),
            "median_transition_acceptance": float(
                np.median([row["transition_acceptance"] for row in selected])
            ),
            "median_objective_improvement_rate": float(
                np.median([row["objective_improvement_rate"] for row in selected])
            ),
            "median_incumbent_update_rate": float(
                np.median([row["incumbent_update_rate"] for row in selected])
            ),
        }
    payload = {
        "schema": "neuralopt.hamiltonian-smoke.v1",
        "environment": {"platform": platform.platform(), "jax": jax.__version__},
        "budget": {"objective_evaluations": 100, "hmc_leapfrog_steps": 2},
        "summary": summary,
        "rows": rows,
    }
    output = root / "results/hamiltonian-smoke-v0.2.0.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
