"""Train and benchmark scale-aware transferable stochastic geometry."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from neuralopt import Problem, minimize
from neuralopt.langevin import OverdampedConfig, UnderdampedConfig
from neuralopt.model import ModelConfig, init_model
from neuralopt.tasks import analytic_task, quadratic_task
from neuralopt.training import StochasticTrainingConfig, load_checkpoint, save_checkpoint, train


def scaled_task(
    family: str,
    seed: int,
    dimension: int,
    objective_scale: float,
    coordinate_scale: float,
) -> Problem:
    if family == "isotropic":
        base = quadratic_task(jax.random.PRNGKey(seed), dimension, 1.0)
    elif family == "ill_conditioned":
        base = quadratic_task(jax.random.PRNGKey(seed), dimension, 1e3)
    elif family == "rotated":
        base = quadratic_task(jax.random.PRNGKey(seed), dimension, 100.0, True)
    else:
        base = analytic_task(family, dimension, seed)
    lo, hi = base.bounds

    def objective(x: Any) -> Any:
        return objective_scale * base.objective(x / coordinate_scale)

    return Problem(
        objective,
        (lo * coordinate_scale, hi * coordinate_scale),
        dimension,
        identifier=f"scaled_{family}",
        x0=base.x0 * coordinate_scale,
        metadata={
            **base.metadata,
            "known_optimum": objective_scale * float(base.metadata["known_optimum"]),
            "objective_scale": objective_scale,
            "coordinate_scale": coordinate_scale,
        },
    )


def task_set(seeds: tuple[int, ...], dimensions: tuple[int, ...]) -> list[tuple[str, int, Problem]]:
    families = ("isotropic", "ill_conditioned", "rotated", "rastrigin", "ackley")
    rows = []
    for index, seed in enumerate(seeds):
        for family_index, family in enumerate(families):
            dimension = dimensions[(index + family_index) % len(dimensions)]
            objective_scale = (0.2, 1.0, 5.0)[(index + family_index) % 3]
            coordinate_scale = (0.5, 1.0, 2.0)[(2 * index + family_index) % 3]
            rows.append(
                (
                    family,
                    seed,
                    scaled_task(
                        family,
                        seed + 100 * family_index,
                        dimension,
                        objective_scale,
                        coordinate_scale,
                    ),
                )
            )
    return rows


def bootstrap(values: list[float], seed: int = 0) -> list[float]:
    data = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.choice(data, (10_000, data.size), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def stochastic_config(
    integrator: str, structure: str, fp_weight: float = 0.0
) -> StochasticTrainingConfig:
    return StochasticTrainingConfig(
        integrator=integrator,  # type: ignore[arg-type]
        episodes=8,
        integration_steps=2,
        horizons=(2, 4, 8),
        step_size=0.008,
        temperature=0.5,
        friction=1.0,
        stability_weight=1e-3,
        fp_weight=fp_weight,
        geometry_structure=structure,  # type: ignore[arg-type]
        gradient_samples=4,
        temperature_range=(0.25, 1.0),
        friction_range=(0.5, 2.0),
    )


def evaluate(
    problem: Problem,
    seed: int,
    integrator: str,
    variant: str,
    params: Any,
    config: ModelConfig,
    geometry: str,
    structure: str,
    fixed_scale: float,
    budget: int = 100,
    temperature: float = 0.5,
) -> dict[str, Any]:
    method = "langevin" if integrator == "overdamped" else "underdamped"
    kwargs = dict(
        problem=problem,
        max_evaluations=budget,
        seed=seed,
        params=params,
        config=config,
        candidate_method=method,
        hamiltonian_metric=geometry,
        fixed_metric_diag=np.full(problem.dimension, fixed_scale),
        learned_metric_structure=structure,
        overdamped_config=OverdampedConfig(step_size=0.008, temperature=temperature, steps=2),
        underdamped_config=UnderdampedConfig(
            step_size=0.008, temperature=temperature, friction=1.0, steps=2
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
    records = diagnostics["records"]
    optimum = float(problem.metadata["known_optimum"])
    initial_regret = max(float(result.trajectory[0]) - optimum, 0.0)
    target = 0.1 * max(initial_regret, 1e-12)
    target_indices = [
        index for index, value in enumerate(result.trajectory) if value - optimum <= target
    ]
    return {
        "integrator": integrator,
        "variant": variant,
        "seed": seed,
        "dimension": problem.dimension,
        "objective_scale": problem.metadata["objective_scale"],
        "coordinate_scale": problem.metadata["coordinate_scale"],
        "budget": budget,
        "temperature": temperature,
        "best_objective": result.fun,
        "log_regret": float(np.log1p(max(result.fun - optimum, 0.0))),
        "anytime_log_regret_auc": float(
            np.mean(np.log1p(np.maximum(np.asarray(result.trajectory) - optimum, 0.0)))
        ),
        "success": bool(target_indices),
        "time_to_target_episode": min(target_indices) if target_indices else None,
        "objective_evaluations": diagnostics["objective_evaluations"],
        "gradient_evaluations": diagnostics["gradient_evaluations"],
        "integration_steps": diagnostics["stochastic_integration_steps"],
        "candidate_improvement_rate": diagnostics["objective_improvement_rate"],
        "incumbent_update_rate": diagnostics["incumbent_update_rate"],
        "median_condition": float(np.median([row["preconditioner_condition"] for row in records])),
        "median_lowrank_norm": float(np.median([row["lowrank_frobenius"] for row in records])),
        "median_displacement": float(
            np.median([row["normalized_candidate_distance"] for row in records])
        ),
        "finite_rate": 1.0 - float(np.mean([row["reason"] == "nonfinite" for row in records])),
        "first_seconds": first_seconds,
        "warm_seconds": warm_seconds,
    }


def main(reuse_checkpoints: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    config = ModelConfig(
        width=16,
        heads=4,
        rank=2,
        layers=1,
        scale_aware=True,
        geometry_bound=4.0,
        lowrank_scale=0.5,
    )
    train_rows = task_set((0, 1), (2, 4, 6))
    validation_rows = task_set((20, 21), (3, 5))
    test_rows = task_set((100, 101, 102, 103, 104, 105), (7, 8, 10))
    variants = {
        "overdamped_diagonal": stochastic_config("overdamped", "diagonal"),
        "overdamped_lowrank": stochastic_config("overdamped", "lowrank"),
        "overdamped_full": stochastic_config("overdamped", "full"),
        "overdamped_fp": stochastic_config("overdamped", "full", 0.05),
        "underdamped_diagonal": stochastic_config("underdamped", "diagonal"),
        "underdamped_full": stochastic_config("underdamped", "full"),
    }
    result_path = root / "results/transfer-geometry-v0.5.0.json"
    previous = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if reuse_checkpoints and result_path.exists()
        else None
    )
    checkpoints = {}
    training = {}
    for name, training_config in variants.items():
        path = root / f"artifacts/neuralopt-v0.5.0-{name}.npz"
        if reuse_checkpoints:
            params, loaded = load_checkpoint(path)
            if loaded != config:
                raise ValueError(f"configuration mismatch for {name}")
            elapsed = previous["training"][name]["seconds"]
            losses = []
        else:
            start = time.perf_counter()
            params, losses = train(
                [row[2] for row in train_rows],
                seed=31,
                steps=144,
                learning_rate=3e-4,
                config=config,
                objective="stochastic",
                stochastic=training_config,
            )
            elapsed = time.perf_counter() - start
            save_checkpoint(
                path,
                params,
                config,
                {"seed": 31, "updates": 144, "training_config": repr(training_config)},
            )
        checkpoints[name] = params
        training[name] = {
            "seconds": elapsed,
            "updates": 144,
            "initial_loss": losses[0] if losses else previous["training"][name]["initial_loss"],
            "final_loss": losses[-1] if losses else previous["training"][name]["final_loss"],
            "checkpoint_bytes": path.stat().st_size,
        }

    # A short-budget checkpoint isolates insufficient meta-optimization from
    # representational limitations without selecting on the final test set.
    short_path = root / "artifacts/neuralopt-v0.5.0-overdamped_full_36.npz"
    if reuse_checkpoints:
        short_params, _ = load_checkpoint(short_path)
        training["overdamped_full_36"] = previous["training"]["overdamped_full_36"]
    else:
        short_params, short_losses = train(
            [row[2] for row in train_rows],
            seed=31,
            steps=36,
            learning_rate=3e-4,
            config=config,
            objective="stochastic",
            stochastic=variants["overdamped_full"],
        )
        save_checkpoint(short_path, short_params, config, {"seed": 31, "updates": 36})
        training["overdamped_full_36"] = {
            "seconds": None,
            "updates": 36,
            "initial_loss": short_losses[0],
            "final_loss": short_losses[-1],
            "checkpoint_bytes": short_path.stat().st_size,
        }
    checkpoints["overdamped_full_36"] = short_params

    untrained = init_model(jax.random.PRNGKey(31), config)
    fixed_candidates = (0.25, 0.5, 1.0, 2.0, 4.0)
    fixed_tuning = {}
    for integrator in ("overdamped", "underdamped"):
        scores = {}
        for fixed_scale in fixed_candidates:
            values = []
            for _family, seed, problem in validation_rows:
                row = evaluate(
                    problem,
                    seed,
                    integrator,
                    f"fixed_{fixed_scale}",
                    untrained,
                    config,
                    "fixed",
                    "full",
                    fixed_scale,
                    budget=80,
                )
                values.append(row["log_regret"])
            scores[str(fixed_scale)] = float(np.mean(values))
        fixed_tuning[integrator] = {
            "scores": scores,
            "selected": float(min(scores, key=scores.get)),
        }

    evaluation_rows = []
    for integrator in ("overdamped", "underdamped"):
        learned = (
            (
                "learned_diagonal",
                checkpoints[f"{integrator}_diagonal"],
                "diagonal",
            ),
            ("learned_full", checkpoints[f"{integrator}_full"], "full"),
        )
        if integrator == "overdamped":
            learned += (
                ("learned_lowrank", checkpoints["overdamped_lowrank"], "lowrank"),
                ("learned_fp", checkpoints["overdamped_fp"], "full"),
                ("learned_full_36", checkpoints["overdamped_full_36"], "full"),
            )
        geometry_variants = (
            ("identity", untrained, "identity", "full"),
            ("fixed_tuned", untrained, "fixed", "full"),
            ("untrained", untrained, "learned", "full"),
        ) + tuple((name, params, "learned", structure) for name, params, structure in learned)
        for family, seed, problem in test_rows:
            for variant, params, geometry, structure in geometry_variants:
                row = evaluate(
                    problem,
                    seed,
                    integrator,
                    variant,
                    params,
                    config,
                    geometry,
                    structure,
                    fixed_tuning[integrator]["selected"],
                )
                row["family"] = family
                evaluation_rows.append(row)

    summary = {}
    family_summary = {}
    for integrator in ("overdamped", "underdamped"):
        selected = [row for row in evaluation_rows if row["integrator"] == integrator]
        fixed = {
            (row["family"], row["seed"]): row["log_regret"]
            for row in selected
            if row["variant"] == "fixed_tuned"
        }
        for variant in sorted({row["variant"] for row in selected}):
            group = [row for row in selected if row["variant"] == variant]
            differences = [row["log_regret"] - fixed[(row["family"], row["seed"])] for row in group]
            summary[f"{integrator}:{variant}"] = {
                "mean_log_regret": float(np.mean([row["log_regret"] for row in group])),
                "mean_anytime_auc": float(
                    np.mean([row["anytime_log_regret_auc"] for row in group])
                ),
                "difference_vs_fixed": float(np.mean(differences)),
                "paired_95_percent_ci": bootstrap(differences),
                "wins_vs_fixed": int(np.sum(np.asarray(differences) < 0)),
                "success_rate": float(np.mean([row["success"] for row in group])),
                "median_warm_seconds": float(np.median([row["warm_seconds"] for row in group])),
                "median_condition": float(np.median([row["median_condition"] for row in group])),
                "finite_rate": float(np.mean([row["finite_rate"] for row in group])),
            }
            for family in sorted({row["family"] for row in group}):
                family_group = [row for row in group if row["family"] == family]
                family_summary[f"{integrator}:{variant}:{family}"] = {
                    "mean_log_regret": float(np.mean([row["log_regret"] for row in family_group])),
                    "instances": len(family_group),
                }

    # Frozen-checkpoint transfer outside the training temperature range and at
    # shorter/longer inference horizons. The fixed geometry is not retuned.
    generalization_rows = []
    for integrator in ("overdamped", "underdamped"):
        variants_for_transfer = (
            ("fixed_tuned", untrained, "fixed", "full"),
            (
                "learned_diagonal",
                checkpoints[f"{integrator}_diagonal"],
                "learned",
                "diagonal",
            ),
            ("learned_full", checkpoints[f"{integrator}_full"], "learned", "full"),
        )
        if integrator == "overdamped":
            variants_for_transfer += (
                ("learned_fp", checkpoints["overdamped_fp"], "learned", "full"),
            )
        for temperature in (0.125, 1.5):
            for budget in (50, 150):
                for family, seed, problem in test_rows:
                    for variant, params, geometry, structure in variants_for_transfer:
                        row = evaluate(
                            problem,
                            seed,
                            integrator,
                            variant,
                            params,
                            config,
                            geometry,
                            structure,
                            fixed_tuning[integrator]["selected"],
                            budget=budget,
                            temperature=temperature,
                        )
                        row["family"] = family
                        generalization_rows.append(row)

    generalization_summary = {}
    settings = sorted(
        {(row["integrator"], row["temperature"], row["budget"]) for row in generalization_rows}
    )
    for integrator, temperature, budget in settings:
        setting_rows = [
            row
            for row in generalization_rows
            if row["integrator"] == integrator
            and row["temperature"] == temperature
            and row["budget"] == budget
        ]
        fixed = {
            (row["family"], row["seed"]): row["log_regret"]
            for row in setting_rows
            if row["variant"] == "fixed_tuned"
        }
        for variant in sorted({row["variant"] for row in setting_rows}):
            group = [row for row in setting_rows if row["variant"] == variant]
            differences = [row["log_regret"] - fixed[(row["family"], row["seed"])] for row in group]
            key = f"{integrator}:T={temperature}:budget={budget}:{variant}"
            generalization_summary[key] = {
                "mean_log_regret": float(np.mean([row["log_regret"] for row in group])),
                "difference_vs_fixed": float(np.mean(differences)),
                "paired_95_percent_ci": bootstrap(differences),
                "wins_vs_fixed": int(np.sum(np.asarray(differences) < 0)),
            }

    payload = {
        "schema": "neuralopt.transfer-geometry.v0.5",
        "environment": {"platform": platform.platform(), "jax": jax.__version__},
        "protocol": {
            "training_instances": len(train_rows),
            "validation_instances": len(validation_rows),
            "test_instances": len(test_rows),
            "test_seeds": [100, 101, 102, 103, 104, 105],
            "test_dimensions": [7, 8, 10],
            "bootstrap_draws": 10_000,
            "objective_budget": 100,
            "training_temperature_range": [0.25, 1.0],
            "training_friction_range": [0.5, 2.0],
        },
        "fixed_tuning": fixed_tuning,
        "training": training,
        "summary": summary,
        "family_summary": family_summary,
        "rows": evaluation_rows,
        "generalization_summary": generalization_summary,
        "generalization_rows": generalization_rows,
    }
    output = root / "results/transfer-geometry-v0.5.0.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixed_tuning": fixed_tuning, "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-checkpoints", action="store_true")
    main(parser.parse_args().reuse_checkpoints)
