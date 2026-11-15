"""CPU-capable supervised policy training and portable checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from .features import gradient_to_normalized, parameter_features, safe_l2_norm
from .model import (
    DEFAULT_CONFIG,
    ModelConfig,
    Params,
    apply_mlp_model,
    apply_model,
    init_mlp_model,
    init_model,
    precondition,
)
from .problem import Problem


def imitation_loss(
    params: Params,
    problem: Problem,
    key: Any,
    config: ModelConfig,
    model_kind: str = "transformer",
) -> Any:
    """Train directions toward known improving quadratic-style displacement."""
    lo, hi = problem.bounds
    x = jax.random.uniform(key, (problem.dimension,), minval=lo, maxval=hi)
    grad = jax.grad(problem.objective)(x)
    apply = apply_model if model_kind == "transformer" else apply_mlp_model
    out = apply(params, parameter_features(x, grad, lo, hi, problem.x0), 1.0, config)
    target = -grad / safe_l2_norm(grad)
    return jnp.mean((out["direction"] - target) ** 2) + 1e-4 * jnp.mean(out["lowrank"] ** 2)


@dataclass(frozen=True)
class UnrollConfig:
    """Static differentiable optimization and loss configuration."""

    steps: int = 25
    horizons: tuple[int, ...] = (5, 10, 25)
    stability_weight: float = 1e-3
    direction_weight: float = 1e-2
    preconditioner_weight: float = 1e-4


def unrolled_regret_loss(
    params: Params,
    problem: Problem,
    key: Any,
    config: ModelConfig,
    unroll: UnrollConfig = UnrollConfig(),
    model_kind: str = "transformer",
) -> Any:
    """Differentiate best-so-far multi-horizon regret through learned updates.

    The training path uses a continuous projected update. Hard inference
    acceptance and branch decisions are intentionally absent; gradients through
    ``minimum`` are valid subgradients of the incumbent objective.
    """
    lo, hi = problem.bounds
    x0 = jax.random.uniform(key, (problem.dimension,), minval=lo, maxval=hi)
    f0 = problem.objective(x0)
    if "known_optimum" not in problem.metadata:
        raise ValueError("regret training requires problem.metadata['known_optimum']")
    optimum = jnp.asarray(problem.metadata["known_optimum"], dtype=f0.dtype)
    scale = jnp.maximum(jnp.abs(f0 - optimum), 1.0)

    def update(carry: tuple[Any, Any], t: Any) -> tuple[tuple[Any, Any], tuple[Any, ...]]:
        x, best = carry
        grad = jax.grad(problem.objective)(x)
        apply = apply_model if model_kind == "transformer" else apply_mlp_model
        out = apply(
            params,
            parameter_features(x, grad, lo, hi, x),
            1.0 - t / unroll.steps,
            config,
        )
        raw = out["direction"]
        direction = precondition(raw, out["diag"], out["lowrank"])
        unit = direction / safe_l2_norm(direction)
        candidate = jnp.clip(x + 0.25 * out["step"] * (hi - lo) * unit, lo, hi)
        value = problem.objective(candidate)
        new_best = jnp.minimum(best, value)
        normalized_grad = gradient_to_normalized(grad, lo, hi)
        alignment = jnp.sum(unit * normalized_grad) / safe_l2_norm(normalized_grad)
        regularizer = (
            unroll.stability_weight * jnp.mean((candidate - x) ** 2 / ((hi - lo) ** 2))
            + unroll.direction_weight * jax.nn.relu(alignment)
            + unroll.preconditioner_weight * jnp.mean(jnp.log(out["diag"]) ** 2)
        )
        return (candidate, new_best), (new_best, regularizer)

    (_, _), (best_values, regularizers) = jax.lax.scan(
        update, (x0, f0), jnp.arange(unroll.steps, dtype=f0.dtype)
    )
    indices = jnp.asarray([min(h, unroll.steps) - 1 for h in unroll.horizons])
    regrets = jnp.maximum(best_values[indices] - optimum, 0.0) / scale
    weights = 1.0 / jnp.asarray(unroll.horizons, dtype=f0.dtype)
    weights = weights / jnp.sum(weights)
    return jnp.sum(weights * jnp.log1p(regrets)) + jnp.mean(regularizers)


def train(
    problems: Iterable[Problem],
    seed: int = 0,
    steps: int = 100,
    learning_rate: float = 1e-3,
    config: ModelConfig = DEFAULT_CONFIG,
    objective: str = "imitation",
    unroll: UnrollConfig = UnrollConfig(),
    model_kind: str = "transformer",
) -> tuple[Params, list[float]]:
    """Train with Optax; objective optimization inference never uses Optax."""
    try:
        import optax
    except ImportError as exc:
        raise ImportError("install neuralopt[train] to train") from exc
    items = list(problems)
    if not items:
        raise ValueError("at least one training problem is required")
    if objective not in {"imitation", "regret"}:
        raise ValueError("objective must be 'imitation' or 'regret'")
    key = jax.random.PRNGKey(seed)
    if model_kind not in {"transformer", "mlp"}:
        raise ValueError("model_kind must be 'transformer' or 'mlp'")
    params = init_model(key, config) if model_kind == "transformer" else init_mlp_model(key, config)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    state = optimizer.init(params)

    def compiled_loss(problem: Problem) -> Any:
        if objective == "imitation":

            def task_loss(p: Params, k: Any) -> Any:
                return imitation_loss(p, problem, k, config, model_kind)
        else:

            def task_loss(p: Params, k: Any) -> Any:
                return unrolled_regret_loss(p, problem, k, config, unroll, model_kind)

        return jax.jit(jax.value_and_grad(task_loss))

    loss_and_grad = [compiled_loss(problem) for problem in items]
    losses: list[float] = []
    for step in range(steps):
        key, sub = jax.random.split(key)
        loss, grads = loss_and_grad[step % len(items)](params, sub)
        if not bool(jnp.isfinite(loss)):
            raise FloatingPointError(f"nonfinite training loss at step {step}")
        gradients_finite = all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(grads)
        )
        if not gradients_finite:
            raise FloatingPointError(f"nonfinite training gradient at step {step}")
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
