"""Matched-budget direct, HMC, and Langevin optimization smoke study."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import jax
import numpy as np

from neuralopt import minimize
from neuralopt.hamiltonian import HamiltonianConfig
from neuralopt.langevin import OverdampedConfig, UnderdampedConfig
from neuralopt.tasks import analytic_task, quadratic_task
from neuralopt.training import load_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    params, model_config = load_checkpoint(root / "artifacts/demo-v0.1.0.npz")
    variants = (
        ("direct", "direct", "learned"),
        ("hmc_identity", "hamiltonian", "identity"),
        ("hmc_learned", "hamiltonian", "learned"),
        ("overdamped_identity", "langevin", "identity"),
        ("overdamped_learned", "langevin", "learned"),
        ("underdamped_identity", "underdamped", "identity"),
        ("underdamped_learned", "underdamped", "learned"),
    )
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
            for algorithm, method, geometry in variants:
                kwargs = {
                    "problem": problem,
                    "max_evaluations": 100,
                    "seed": seed,
                    "params": params,
                    "config": model_config,
                    "candidate_method": method,
                    "hamiltonian_metric": geometry,
                    "hamiltonian_config": HamiltonianConfig(step_size=0.03, leapfrog_steps=2),
                    "overdamped_config": OverdampedConfig(step_size=0.01, temperature=0.5, steps=2),
                    "underdamped_config": UnderdampedConfig(
                        step_size=0.02, temperature=0.5, friction=1.0, steps=2
                    ),
                    "record_diagnostics": True,
                }
                # Compile once, then report warm execution separately.
                start = time.perf_counter()
                minimize(**kwargs)
                first_call = time.perf_counter() - start
                start = time.perf_counter()
                result = minimize(**kwargs)
                warm = time.perf_counter() - start
                diagnostics = result.diagnostics or {}
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "dimension": problem.dimension,
                        "algorithm": algorithm,
                        "best_objective": result.fun,
                        "first_call_seconds": first_call,
                        "warm_runtime_seconds": warm,
                        "objective_evaluations": diagnostics["objective_evaluations"],
                        "gradient_evaluations": diagnostics["gradient_evaluations"],
                        "candidate_proposals": diagnostics["candidate_proposals"],
                        "objective_improvement_rate": diagnostics["objective_improvement_rate"],
                        "incumbent_update_rate": diagnostics["incumbent_update_rate"],
                        "integration_steps": diagnostics["stochastic_integration_steps"],
                        "leapfrog_steps": diagnostics["leapfrog_steps"],
                    }
                )
    summary = {}
    for algorithm in sorted({row["algorithm"] for row in rows}):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        summary[algorithm] = {
            "median_objective": float(np.median([r["best_objective"] for r in selected])),
            "median_warm_seconds": float(np.median([r["warm_runtime_seconds"] for r in selected])),
            "median_first_call_seconds": float(
                np.median([r["first_call_seconds"] for r in selected])
            ),
            "median_improvement_rate": float(
                np.median([r["objective_improvement_rate"] for r in selected])
            ),
            "median_incumbent_rate": float(
                np.median([r["incumbent_update_rate"] for r in selected])
            ),
        }
    payload = {
        "schema": "neuralopt.stochastic-smoke.v0.3",
        "environment": {"platform": platform.platform(), "jax": jax.__version__},
        "summary": summary,
        "rows": rows,
    }
    output = root / "results/stochastic-smoke-v0.3.0.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
