"""Deterministic legacy (v0.1--v0.5) outputs used as a frozen regression contract.

``tests/data/legacy-v0.5-golden.json`` was generated from these cases with the
unmodified v0.5.0 source (commit 90623b7) before any v0.6 change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuralopt import Problem, minimize
from neuralopt.langevin import OverdampedConfig, UnderdampedConfig
from neuralopt.model import apply_model
from neuralopt.tasks import quadratic_task
from neuralopt.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
# Pinned explicitly: later versioned artifacts (e.g. v0.6 history schemas) must not
# silently join the frozen v0.1--v0.5 compatibility contract.
FROZEN_ARTIFACTS = (
    "demo-v0.1.0.npz",
    "neuralopt-v0.2.0-smoke.npz",
    "neuralopt-v0.4.0-regret_baoab.npz",
    "neuralopt-v0.4.0-regret_fp_overdamped.npz",
    "neuralopt-v0.4.0-regret_overdamped.npz",
    "neuralopt-v0.5.0-overdamped_diagonal.npz",
    "neuralopt-v0.5.0-overdamped_fp.npz",
    "neuralopt-v0.5.0-overdamped_full.npz",
    "neuralopt-v0.5.0-overdamped_full_36.npz",
    "neuralopt-v0.5.0-overdamped_lowrank.npz",
    "neuralopt-v0.5.0-underdamped_diagonal.npz",
    "neuralopt-v0.5.0-underdamped_full.npz",
)
CHECKPOINTS = [ROOT / "artifacts" / name for name in FROZEN_ARTIFACTS] + [
    ROOT / "src/neuralopt/checkpoints/demo-v0.1.0.npz"
]


def _rotated(dimension: int) -> Problem:
    return quadratic_task(jax.random.PRNGKey(7), dimension, 100.0, True)


def _result(result: Any) -> dict[str, Any]:
    return {
        "x": np.asarray(result.x, dtype=float).tolist(),
        "fun": float(result.fun),
        "evaluations": int(result.evaluations),
        "trajectory": [float(v) for v in result.trajectory],
    }


def legacy_outputs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["direct_default"] = _result(minimize(_rotated(4), max_evaluations=30, seed=0))
    full_od, cfg_od = load_checkpoint(ROOT / "artifacts/neuralopt-v0.5.0-overdamped_full.npz")
    full_ud, cfg_ud = load_checkpoint(ROOT / "artifacts/neuralopt-v0.5.0-underdamped_full.npz")
    od = OverdampedConfig(step_size=0.008, temperature=0.5, steps=2)
    ud = UnderdampedConfig(step_size=0.008, temperature=0.5, friction=1.0, steps=2)
    out["langevin_learned"] = _result(
        minimize(
            _rotated(5),
            max_evaluations=30,
            seed=1,
            params=full_od,
            config=cfg_od,
            candidate_method="langevin",
            overdamped_config=od,
        )
    )
    out["underdamped_learned"] = _result(
        minimize(
            _rotated(5),
            max_evaluations=31,
            seed=2,
            params=full_ud,
            config=cfg_ud,
            candidate_method="underdamped",
            underdamped_config=ud,
        )
    )
    out["langevin_fixed"] = _result(
        minimize(
            _rotated(5),
            max_evaluations=30,
            seed=3,
            params=full_od,
            config=cfg_od,
            candidate_method="langevin",
            hamiltonian_metric="fixed",
            fixed_metric_diag=np.full(5, 0.25),
            overdamped_config=od,
        )
    )
    out["hamiltonian_identity"] = _result(
        minimize(
            _rotated(3),
            max_evaluations=40,
            seed=4,
            candidate_method="hamiltonian",
            hamiltonian_metric="identity",
        )
    )
    features = jnp.asarray(
        np.linspace(-0.9, 0.8, 15, dtype=np.float32).reshape(3, 5)
    )
    for path in CHECKPOINTS:
        params, config = load_checkpoint(path)
        model = apply_model(params, features, 0.5, config)
        out[f"checkpoint:{path.name}"] = {
            "config": [
                config.width,
                config.heads,
                config.rank,
                config.layers,
                config.scale_aware,
                config.geometry_bound,
                config.lowrank_scale,
            ],
            "param_shapes": {k: list(v.shape) for k, v in sorted(params.items())},
            "outputs": {
                k: np.asarray(v, dtype=float).ravel().tolist() for k, v in sorted(model.items())
            },
        }
    return out
