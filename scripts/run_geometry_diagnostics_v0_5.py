"""Diagnose v0.4/v0.5 learned stochastic geometry failure mechanisms."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
from run_transfer_geometry_v0_5 import evaluate, stochastic_config, task_set

from neuralopt.features import parameter_features
from neuralopt.model import apply_model
from neuralopt.training import load_checkpoint, stochastic_geometry_loss


def geometry_gradient(params: Any, problem: Any, config: Any, training_config: Any, seed: int):
    def loss_fn(candidate_params):
        return stochastic_geometry_loss(
            candidate_params,
            problem,
            jax.random.PRNGKey(seed),
            config,
            training_config,
        )

    gradients = jax.grad(loss_fn)(params)
    return np.concatenate([np.asarray(gradients[name]).ravel() for name in ("diag", "lowrank")])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    v04 = json.loads((root / "results/geometry-training-v0.4.0.json").read_text())
    v04_reproduced = json.loads(
        (root / "results/geometry-training-v0.4.0-reproduced.json").read_text()
    )
    v05 = json.loads((root / "results/transfer-geometry-v0.5.0.json").read_text())
    deterministic_fields = (
        "best_objective",
        "log_regret",
        "anytime_log_regret_auc",
        "objective_evaluations",
        "gradient_evaluations",
        "integration_steps",
        "improvement_rate",
        "finite_rate",
    )
    reproduction_error = {
        field: max(
            abs(float(original[field]) - float(reproduced[field]))
            for original, reproduced in zip(v04["rows"], v04_reproduced["rows"], strict=True)
        )
        for field in deterministic_fields
    }

    train_rows = task_set((0, 1), (2, 4, 6))
    validation_rows = task_set((20, 21), (3, 5))
    split_results = {}
    for integrator, fixed_scale in (("overdamped", 0.25), ("underdamped", 2.0)):
        params, config = load_checkpoint(root / f"artifacts/neuralopt-v0.5.0-{integrator}_full.npz")
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            fixed_values, learned_values = [], []
            for _family, seed, problem in rows:
                fixed_values.append(
                    evaluate(
                        problem,
                        seed,
                        integrator,
                        "fixed",
                        params,
                        config,
                        "fixed",
                        "full",
                        fixed_scale,
                        budget=80,
                    )["log_regret"]
                )
                learned_values.append(
                    evaluate(
                        problem,
                        seed,
                        integrator,
                        "learned",
                        params,
                        config,
                        "learned",
                        "full",
                        fixed_scale,
                        budget=80,
                    )["log_regret"]
                )
            split_results[f"{integrator}:{split}"] = {
                "fixed_mean_log_regret": float(np.mean(fixed_values)),
                "learned_mean_log_regret": float(np.mean(learned_values)),
                "learned_minus_fixed": float(np.mean(np.asarray(learned_values) - fixed_values)),
            }

    params, config = load_checkpoint(root / "artifacts/neuralopt-v0.5.0-overdamped_full.npz")
    single = replace(stochastic_config("overdamped", "full"), gradient_samples=1)
    averaged = replace(single, gradient_samples=4)
    gradient_variance = {}
    for label, training_config in (("one_sample", single), ("four_sample", averaged)):
        coefficients = []
        for _family, _seed, problem in train_rows[:3]:
            for seed in range(6):
                coefficients.append(
                    geometry_gradient(params, problem, config, training_config, seed)
                )
        norms = np.asarray([np.linalg.norm(value) for value in coefficients])
        gradient_variance[label] = {
            "mean_norm": float(norms.mean()),
            "norm_coefficient_of_variation": float(norms.std() / norms.mean()),
        }

    old_params, old_config = load_checkpoint(
        root / "artifacts/neuralopt-v0.4.0-regret_overdamped.npz"
    )
    problem = train_rows[0][2]
    lo, hi = problem.bounds
    x = problem.x0
    gradient = jax.grad(problem.objective)(x)

    def scale_delta(candidate_params, candidate_config):
        first = apply_model(
            candidate_params,
            parameter_features(x, gradient, lo, hi, x, candidate_config.scale_aware),
            1.0,
            candidate_config,
        )["diag"]
        second = apply_model(
            candidate_params,
            parameter_features(x, 1e3 * gradient, lo, hi, x, candidate_config.scale_aware),
            1.0,
            candidate_config,
        )["diag"]
        return float(np.max(np.abs(np.asarray(second - first))))

    learned_rows = [row for row in v05["rows"] if row["variant"].startswith("learned")]
    payload = {
        "schema": "neuralopt.geometry-diagnostics.v0.5",
        "v0.4_reproduction_max_absolute_error": reproduction_error,
        "v0.4_discrepancy": {
            "pre_review_commit": "c9ee283",
            "pre_review_fp_minus_regret": -0.08895028920757624,
            "pre_review_95_percent_ci": [-0.18076421836224646, -0.008140756169652973],
            "corrected_commit": "0ecc22f",
            "corrected_fp_minus_regret": v04["summary"]["overdamped:fp_effect_vs_regret"][
                "paired_difference"
            ],
            "corrected_95_percent_ci": v04["summary"]["overdamped:fp_effect_vs_regret"][
                "paired_95_percent_ci"
            ],
            "population_each": 10,
            "explanation": (
                "The population was unchanged. Review fixes removed quadratic-noise variance "
                "from the FP loss and matched BAOAB initial momentum, then retrained checkpoints."
            ),
        },
        "objective_scale_diag_response": {
            "v0.4_max_delta_for_1000x_gradient": scale_delta(old_params, old_config),
            "v0.5_max_delta_for_1000x_gradient": scale_delta(params, config),
        },
        "meta_gradient_variance": gradient_variance,
        "split_regret": split_results,
        "learned_geometry": {
            "median_condition": float(np.median([row["median_condition"] for row in learned_rows])),
            "max_condition": float(np.max([row["median_condition"] for row in learned_rows])),
            "median_lowrank_norm": float(
                np.median([row["median_lowrank_norm"] for row in learned_rows])
            ),
            "median_displacement": float(
                np.median([row["median_displacement"] for row in learned_rows])
            ),
        },
    }
    output = root / "results/geometry-diagnostics-v0.5.0.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
