"""CPU-capable supervised policy training and portable checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from .features import parameter_features
from .model import DEFAULT_CONFIG, ModelConfig, Params, apply_model, init_model
from .problem import Problem


def imitation_loss(params: Params, problem: Problem, key: Any, config: ModelConfig) -> Any:
    """Train directions toward known improving quadratic-style displacement."""
    lo, hi = problem.bounds
    x = jax.random.uniform(key, (problem.dimension,), minval=lo, maxval=hi)
    grad = jax.grad(problem.objective)(x)
    out = apply_model(params, parameter_features(x, grad, lo, hi, problem.x0), 1.0, config)
    target = -grad / (jnp.linalg.norm(grad) + 1e-8)
    return jnp.mean((out["direction"] - target) ** 2) + 1e-4 * jnp.mean(out["lowrank"] ** 2)


def train(
    problems: Iterable[Problem],
    seed: int = 0,
    steps: int = 100,
    learning_rate: float = 1e-3,
    config: ModelConfig = DEFAULT_CONFIG,
) -> tuple[Params, list[float]]:
    """Train with Optax; objective optimization inference never uses Optax."""
    try:
        import optax
    except ImportError as exc:
        raise ImportError("install neuralopt[train] to train") from exc
    items = list(problems)
    if not items:
        raise ValueError("at least one training problem is required")
    key = jax.random.PRNGKey(seed)
    params = init_model(key, config)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    state = optimizer.init(params)
    losses: list[float] = []
    for step in range(steps):
        key, sub = jax.random.split(key)
        problem = items[step % len(items)]
        loss, grads = jax.value_and_grad(imitation_loss)(params, problem, sub, config)
        updates, state = optimizer.update(grads, state, params)
        params = cast(Params, optax.apply_updates(params, updates))
        losses.append(float(loss))
    return params, losses


def save_checkpoint(
    path: str | Path, params: Params, config: ModelConfig, metadata: dict[str, Any] | None = None
) -> None:
    arrays = {f"param__{k}": np.asarray(v) for k, v in params.items()}
    arrays.update(
        {
            "config": np.array([config.width, config.heads, config.rank, config.layers]),
            "metadata": np.array([repr(metadata or {})]),
        }
    )
    saver: Any = np.savez_compressed
    saver(path, **arrays)


def load_checkpoint(path: str | Path) -> tuple[Params, ModelConfig]:
    with np.load(path, allow_pickle=False) as data:
        c = data["config"].tolist()
        config = ModelConfig(*c)
        params = {
            k.removeprefix("param__"): jnp.asarray(data[k])
            for k in data.files
            if k.startswith("param__")
        }
    return params, config
