"""Train and evaluate v0.4 stochastic-geometry ablations on CPU."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from neuralopt import minimize
from neuralopt.langevin import OverdampedConfig, UnderdampedConfig
from neuralopt.model import ModelConfig, init_model
from neuralopt.tasks import analytic_task, quadratic_task
from neuralopt.training import StochasticTrainingConfig, load_checkpoint, save_checkpoint, train


def training_tasks() -> list[Any]:
    return [
        quadratic_task(jax.random.PRNGKey(1), 2, 10.0),
        quadratic_task(jax.random.PRNGKey(2), 4, 100.0, True),
        quadratic_task(jax.random.PRNGKey(3), 4, 1_000.0),
        analytic_task("rastrigin", 2, 4),
        analytic_task("ackley", 4, 5),
        analytic_task("rosenbrock", 3, 6),
    ]


def held_out_tasks() -> list[tuple[str, int, Any]]:
    result = []
    for seed in (40, 41):
        result.extend(
            [
                (
                    "rotated_quadratic_3d",
                    seed,
                    quadratic_task(jax.random.PRNGKey(seed), 3, 30, True),
                ),
                ("ill_conditioned_5d", seed, quadratic_task(jax.random.PRNGKey(seed + 10), 5, 1e3)),
                ("rosenbrock_4d", seed, analytic_task("rosenbrock", 4, seed)),
                ("ackley_6d", seed, analytic_task("ackley", 6, seed)),
                ("rastrigin_3d", seed, analytic_task("rastrigin", 3, seed)),
            ]
        )
    return result


def confidence_interval(values: list[float], seed: int = 0) -> list[float]:
    data = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.choice(data, (10_000, data.size), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main(reuse_checkpoints: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    model_config = ModelConfig(width=16, heads=4, rank=2, layers=1)
    base = StochasticTrainingConfig(
        episodes=6,
        integration_steps=2,
        horizons=(2, 4, 6),
        step_size=0.008,
        temperature=0.5,
        stability_weight=1e-3,
    )
    tasks = training_tasks()
    training = {}
    checkpoints = {}
    settings = {
        "regret_overdamped": base,
        "regret_baoab": StochasticTrainingConfig(
            integrator="underdamped",
            episodes=6,
            integration_steps=2,
            horizons=(2, 4, 6),
            step_size=0.008,
            temperature=0.5,
        ),
        "regret_fp_overdamped": StochasticTrainingConfig(**{**base.__dict__, "fp_weight": 0.1}),
    }
    previous_result = root / "results/geometry-training-v0.4.0.json"
    previous = (
        json.loads(previous_result.read_text(encoding="utf-8"))
        if reuse_checkpoints and previous_result.exists()
        else None
    )
    for name, stochastic in settings.items():
        path = root / f"artifacts/neuralopt-v0.4.0-{name}.npz"
        if reuse_checkpoints:
            params, loaded_config = load_checkpoint(path)
            if loaded_config != model_config:
                raise ValueError(f"checkpoint configuration mismatch for {name}")
            training[name] = previous["training"][name]
        else:
            start = time.perf_counter()
            params, losses = train(
                tasks,
                seed=17,
                steps=36,
                learning_rate=3e-4,
                config=model_config,
                objective="stochastic",
                stochastic=stochastic,
            )
            elapsed = time.perf_counter() - start
            save_checkpoint(
                path,
                params,
                model_config,
                {
                    "seed": 17,
                    "updates": 36,
                    "training_families": 6,
                    "config": repr(stochastic),
                },
            )
            training[name] = {
                "seconds": elapsed,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "median_loss": float(np.median(losses)),
            }
        checkpoints[name] = params

    untrained = init_model(jax.random.PRNGKey(17), model_config)
    variants = ("identity", "fixed_diagonal", "untrained", "regret_trained", "regret_fp_trained")
    rows = []
    for integrator in ("overdamped", "underdamped"):
        method = "langevin" if integrator == "overdamped" else "underdamped"
        trained = checkpoints["regret_overdamped" if integrator == "overdamped" else "regret_baoab"]
        for family, seed, problem in held_out_tasks():
            for variant in variants:
                if variant == "regret_trained":
                    params, geometry = trained, "learned"
                elif variant == "regret_fp_trained":
                    params, geometry = checkpoints["regret_fp_overdamped"], "learned"
                elif variant == "untrained":
                    params, geometry = untrained, "learned"
                else:
                    params = untrained
                    geometry = "identity" if variant == "identity" else "fixed"
                kwargs = dict(
                    problem=problem,
                    max_evaluations=80,
                    seed=seed,
                    params=params,
                    config=model_config,
                    candidate_method=method,
                    hamiltonian_metric=geometry,
                    fixed_metric_diag=np.geomspace(0.5, 1.5, problem.dimension),
                    overdamped_config=OverdampedConfig(step_size=0.008, temperature=0.5, steps=2),
                    underdamped_config=UnderdampedConfig(
                        step_size=0.008, temperature=0.5, friction=1.0, steps=2
                    ),
                    record_diagnostics=True,
                )
                start = time.perf_counter()
                minimize(**kwargs)
                first_seconds = time.perf_counter() - start
                start = time.perf_counter()
                result = minimize(**kwargs)
                warm_seconds = time.perf_counter() - start
                diagnostics = result.diagnostics or {}
                optimum = float(problem.metadata["known_optimum"])
                anytime = np.maximum(np.asarray(result.trajectory) - optimum, 0.0)
                rows.append(
                    {
                        "integrator": integrator,
                        "variant": variant,
                        "family": family,
                        "seed": seed,
                        "dimension": problem.dimension,
                        "temperature": 0.5,
                        "budget": 80,
                        "best_objective": result.fun,
                        "log_regret": float(np.log1p(max(result.fun - optimum, 0.0))),
                        "anytime_log_regret_auc": float(np.mean(np.log1p(anytime))),
                        "objective_evaluations": diagnostics["objective_evaluations"],
                        "gradient_evaluations": diagnostics["gradient_evaluations"],
                        "integration_steps": diagnostics["stochastic_integration_steps"],
                        "improvement_rate": diagnostics["objective_improvement_rate"],
                        "finite_rate": 1.0
                        - float(
                            np.mean(
                                [r["reason"] == "nonfinite" for r in diagnostics["records"]]
                            )
                        ),
                        "first_seconds": first_seconds,
                        "warm_seconds": warm_seconds,
                    }
                )

    summary = {}
    for integrator in ("overdamped", "underdamped"):
        selected = [row for row in rows if row["integrator"] == integrator]
        identity = {
            (row["family"], row["seed"]): row["log_regret"]
            for row in selected
            if row["variant"] == "identity"
        }
        for variant in variants:
            group = [row for row in selected if row["variant"] == variant]
            differences = [
                row["log_regret"] - identity[(row["family"], row["seed"])] for row in group
            ]
            summary[f"{integrator}:{variant}"] = {
                "mean_log_regret": float(np.mean([row["log_regret"] for row in group])),
                "mean_anytime_auc": float(
                    np.mean([row["anytime_log_regret_auc"] for row in group])
                ),
                "paired_difference_vs_identity": float(np.mean(differences)),
                "paired_95_percent_ci": confidence_interval(differences),
                "wins_vs_identity": int(np.sum(np.asarray(differences) < 0)),
                "median_warm_seconds": float(np.median([row["warm_seconds"] for row in group])),
                "median_first_seconds": float(np.median([row["first_seconds"] for row in group])),
                "mean_gradient_evaluations": float(
                    np.mean([row["gradient_evaluations"] for row in group])
                ),
                "mean_integration_steps": float(
                    np.mean([row["integration_steps"] for row in group])
                ),
                "mean_finite_rate": float(np.mean([row["finite_rate"] for row in group])),
            }
        regret = {
            (row["family"], row["seed"]): row["log_regret"]
            for row in selected
            if row["variant"] == "regret_trained"
        }
        fp_differences = [
            row["log_regret"] - regret[(row["family"], row["seed"])]
            for row in selected
            if row["variant"] == "regret_fp_trained"
        ]
        summary[f"{integrator}:fp_effect_vs_regret"] = {
            "paired_difference": float(np.mean(fp_differences)),
            "paired_95_percent_ci": confidence_interval(fp_differences),
            "wins": int(np.sum(np.asarray(fp_differences) < 0)),
        }

    generalization_rows = []
    for integrator in ("overdamped", "underdamped"):
        method = "langevin" if integrator == "overdamped" else "underdamped"
        regret_params = checkpoints[
            "regret_overdamped" if integrator == "overdamped" else "regret_baoab"
        ]
        for family, seed, problem in held_out_tasks()[:4]:
            for temperature in (0.25, 1.0):
                for budget in (40, 120):
                    for variant, params in (
                        ("regret_trained", regret_params),
                        ("regret_fp_trained", checkpoints["regret_fp_overdamped"]),
                    ):
                        kwargs = dict(
                            problem=problem,
                            max_evaluations=budget,
                            seed=seed,
                            params=params,
                            config=model_config,
                            candidate_method=method,
                            hamiltonian_metric="learned",
                            overdamped_config=OverdampedConfig(
                                step_size=0.008, temperature=temperature, steps=2
                            ),
                            underdamped_config=UnderdampedConfig(
                                step_size=0.008, temperature=temperature, steps=2
                            ),
                        )
                        minimize(**kwargs)
                        start = time.perf_counter()
                        result = minimize(**kwargs)
                        warm = time.perf_counter() - start
                        optimum = float(problem.metadata["known_optimum"])
                        generalization_rows.append(
                            {
                                "integrator": integrator,
                                "variant": variant,
                                "family": family,
                                "seed": seed,
                                "dimension": problem.dimension,
                                "temperature": temperature,
                                "budget": budget,
                                "log_regret": float(
                                    np.log1p(max(result.fun - optimum, 0.0))
                                ),
                                "warm_seconds": warm,
                            }
                        )

    generalization_summary = {}
    for integrator in ("overdamped", "underdamped"):
        for temperature in (0.25, 1.0):
            for budget in (40, 120):
                selected = [
                    row
                    for row in generalization_rows
                    if row["integrator"] == integrator
                    and row["temperature"] == temperature
                    and row["budget"] == budget
                ]
                regret = {
                    (row["family"], row["seed"]): row["log_regret"]
                    for row in selected
                    if row["variant"] == "regret_trained"
                }
                differences = [
                    row["log_regret"] - regret[(row["family"], row["seed"])]
                    for row in selected
                    if row["variant"] == "regret_fp_trained"
                ]
                generalization_summary[f"{integrator}:T={temperature}:budget={budget}"] = {
                    "fp_difference_vs_regret": float(np.mean(differences)),
                    "paired_95_percent_ci": confidence_interval(differences),
                    "fp_wins": int(np.sum(np.asarray(differences) < 0)),
                    "instances": len(differences),
                }

    payload = {
        "schema": "neuralopt.geometry-training.v0.4",
        "environment": {"platform": platform.platform(), "jax": jax.__version__},
        "protocol": {
            "training_instances": 6,
            "held_out_instances": 10,
            "paired_bootstrap_draws": 10_000,
            "primary_temperature": 0.5,
            "primary_objective_budget": 80,
            "generalization_temperatures": [0.25, 1.0],
            "generalization_objective_budgets": [40, 120],
            "note": (
                "first-call timings share process caches and are order-sensitive; "
                "warm timings are primary"
            ),
        },
        "training": training,
        "summary": summary,
        "generalization_summary": generalization_summary,
        "rows": rows,
        "generalization_rows": generalization_rows,
    }
    output = root / "results/geometry-training-v0.4.0.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"training": training, "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-checkpoints", action="store_true")
    main(parser.parse_args().reuse_checkpoints)
