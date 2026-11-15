"""CPU-capable supervised policy training and portable checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from .features import gradient_to_normalized, parameter_features, safe_l2_norm
from .hamiltonian import from_unconstrained, log_abs_det_jacobian, mass_solve, to_unconstrained
from .langevin import mobility_apply, sample_mobility_noise
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
    normalized_grad = gradient_to_normalized(grad, lo, hi)
    target = -normalized_grad / safe_l2_norm(normalized_grad)
    return jnp.mean((out["direction"] - target) ** 2) + 1e-4 * jnp.mean(out["lowrank"] ** 2)


@dataclass(frozen=True)
class UnrollConfig:
    """Static differentiable optimization and loss configuration."""

    steps: int = 25
    horizons: tuple[int, ...] = (5, 10, 25)
    stability_weight: float = 1e-3
    direction_weight: float = 1e-2
    preconditioner_weight: float = 1e-4


@dataclass(frozen=True)
class StochasticTrainingConfig:
    """Differentiable learned-geometry trajectory and loss configuration."""

    integrator: Literal["overdamped", "underdamped"] = "overdamped"
    episodes: int = 8
    integration_steps: int = 2
    horizons: tuple[int, ...] = (2, 4, 8)
    step_size: float = 1e-2
    temperature: float = 0.5
    friction: float = 1.0
    stability_weight: float = 1e-3
    compute_weight: float = 0.0
    fp_weight: float = 0.0
    free_energy_weight: float = 0.0
    transform_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.integrator not in {"overdamped", "underdamped"}:
            raise ValueError("integrator must be 'overdamped' or 'underdamped'")
        if self.episodes < 1 or self.integration_steps < 1:
            raise ValueError("episodes and integration_steps must be positive")
        if not self.horizons or any(h < 1 or h > self.episodes for h in self.horizons):
            raise ValueError("horizons must lie in [1, episodes]")
        if self.step_size <= 0 or self.temperature <= 0 or self.friction <= 0:
            raise ValueError("step_size, temperature, and friction must be positive")
        if min(
            self.stability_weight,
            self.compute_weight,
            self.fp_weight,
            self.free_energy_weight,
        ) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.integrator != "overdamped" and self.fp_weight:
            raise ValueError("weak Fokker--Planck training currently supports overdamped dynamics")


def _quadratic_euler_weak_residual(
    position: Any, drift: Any, covariance_trace: Any, step_size: float
) -> Any:
    """Conditional Euler weak residual for ``phi(z) = mean(z**2)``."""
    drifted = position + step_size * drift
    observed = jnp.mean(drifted**2) - jnp.mean(position**2)
    observed += step_size * covariance_trace / position.size
    generator = 2.0 * jnp.mean(position * drift) + covariance_trace / position.size
    return observed / step_size - generator


def stochastic_geometry_loss(
    params: Params,
    problem: Problem,
    key: Any,
    config: ModelConfig,
    stochastic: StochasticTrainingConfig = StochasticTrainingConfig(),
) -> Any:
    """Differentiate optimization regret through learned stochastic geometry.

    Geometry is history-adaptive between episodes but frozen within each
    integration episode. The weak residual therefore checks the conditional
    frozen-coefficient Euler generator; it is not an invariant-measure claim for
    the complete adaptive policy.
    """
    if "known_optimum" not in problem.metadata:
        raise ValueError("stochastic training requires problem.metadata['known_optimum']")
    lo, hi = problem.bounds
    start_key, trajectory_key = jax.random.split(key)
    x0 = jax.random.uniform(start_key, (problem.dimension,), minval=lo, maxval=hi)
    z0 = to_unconstrained(x0, lo, hi, stochastic.transform_epsilon)
    f0 = problem.objective(x0)
    optimum = jnp.asarray(problem.metadata["known_optimum"], dtype=f0.dtype)
    regret_scale = jnp.maximum(jnp.abs(f0 - optimum), 1.0)
    momentum0 = jnp.zeros_like(z0)
    episode_keys = jax.random.split(trajectory_key, stochastic.episodes)
    damping = jnp.exp(-stochastic.friction * stochastic.step_size)

    def potential(z: Any) -> Any:
        x = from_unconstrained(z, lo, hi)
        return problem.objective(x) - stochastic.temperature * log_abs_det_jacobian(z, lo, hi)

    def episode(
        carry: tuple[Any, Any, Any, Any], inputs: tuple[Any, Any]
    ) -> tuple[tuple[Any, Any, Any, Any], tuple[Any, ...]]:
        z, momentum, best, best_x = carry
        episode_index, episode_key = inputs
        x = from_unconstrained(z, lo, hi)
        gradient = jax.grad(problem.objective)(x)
        out = apply_model(
            params,
            parameter_features(x, gradient, lo, hi, best_x),
            1.0 - episode_index / stochastic.episodes,
            config,
        )
        diag, lowrank = out["diag"], out["lowrank"]
        step_keys = jax.random.split(episode_key, stochastic.integration_steps)
        fp_residual = jnp.asarray(0.0, dtype=f0.dtype)

        if stochastic.integrator == "overdamped":

            def overdamped_step(position: Any, noise_key: Any) -> tuple[Any, Any]:
                potential_gradient = jax.grad(potential)(position)
                drift = -mobility_apply(potential_gradient, diag, lowrank)
                noise = sample_mobility_noise(noise_key, diag, lowrank)
                diffusion = jnp.sqrt(2.0 * stochastic.temperature * stochastic.step_size) * noise
                next_position = position + stochastic.step_size * drift + diffusion
                covariance_trace = 2.0 * stochastic.temperature * (
                    jnp.sum(diag) + jnp.sum(lowrank**2)
                )
                # For the quadratic test function, integrate the Euler noise
                # contribution exactly. A sampled squared noise term has nonzero
                # variance that would otherwise reward collapsing the mobility.
                residual = _quadratic_euler_weak_residual(
                    position, drift, covariance_trace, stochastic.step_size
                )
                return next_position, residual**2

            next_z, residuals = jax.lax.scan(overdamped_step, z, step_keys)
            next_momentum = momentum
            fp_residual = jnp.mean(residuals)
        else:

            def baoab_step(state: tuple[Any, Any], noise_key: Any) -> tuple[tuple[Any, Any], None]:
                position, p = state
                p = p - 0.5 * stochastic.step_size * jax.grad(potential)(position)
                position = position + 0.5 * stochastic.step_size * mass_solve(
                    p, diag, lowrank
                )
                noise_scale = jnp.sqrt(stochastic.temperature * (1.0 - damping**2))
                p = damping * p + noise_scale * sample_mobility_noise(noise_key, diag, lowrank)
                position = position + 0.5 * stochastic.step_size * mass_solve(
                    p, diag, lowrank
                )
                p = p - 0.5 * stochastic.step_size * jax.grad(potential)(position)
                return (position, p), None

            (next_z, next_momentum), _ = jax.lax.scan(baoab_step, (z, momentum), step_keys)

        candidate = from_unconstrained(next_z, lo, hi)
        value = problem.objective(candidate)
        improved = value < best
        next_best = jnp.minimum(best, value)
        next_best_x = jnp.where(improved, candidate, best_x)
        displacement = jnp.mean(((candidate - x) / (hi - lo)) ** 2)
        return (next_z, next_momentum, next_best, next_best_x), (
            next_best,
            displacement,
            fp_residual,
            value,
            candidate,
        )

    indices = jnp.arange(stochastic.episodes, dtype=f0.dtype)
    (_, _, _, _), (best_values, displacements, fp_residuals, values, candidates) = jax.lax.scan(
        episode, (z0, momentum0, f0, x0), (indices, episode_keys)
    )
    horizon_indices = jnp.asarray([h - 1 for h in stochastic.horizons])
    regrets = jnp.maximum(best_values[horizon_indices] - optimum, 0.0) / regret_scale
    weights = 1.0 / jnp.asarray(stochastic.horizons, dtype=f0.dtype)
    regret_loss = jnp.sum(weights * jnp.log1p(regrets)) / jnp.sum(weights)
    # With static episode lengths, compute cost is constant and intentionally has
    # zero parameter gradient. It keeps cost visible without rewarding numerical
    # shortcuts or nondifferentiable early stopping.
    gradient_cost = stochastic.episodes * stochastic.integration_steps
    if stochastic.integrator == "underdamped":
        gradient_cost *= 2
    compute_loss = jnp.asarray(gradient_cost / max(stochastic.horizons), dtype=f0.dtype)
    if stochastic.free_energy_weight:
        # Optional Gaussian entropy approximation over trajectory states. This
        # is a regularizer, not exact free energy for the nonstationary law.
        centered = candidates - jnp.mean(candidates, axis=0)
        covariance = centered.T @ centered / stochastic.episodes
        covariance += 1e-4 * jnp.eye(problem.dimension, dtype=f0.dtype)
        _, logdet = jnp.linalg.slogdet(covariance)
        free_energy_surrogate = jnp.mean(values) - 0.5 * stochastic.temperature * logdet
    else:
        free_energy_surrogate = jnp.asarray(0.0, dtype=f0.dtype)
    return (
        regret_loss
        + stochastic.stability_weight * jnp.mean(displacements)
        + stochastic.compute_weight * compute_loss
        + stochastic.fp_weight * jnp.mean(jnp.log1p(fp_residuals))
        + stochastic.free_energy_weight * free_energy_surrogate
    )


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
    stochastic: StochasticTrainingConfig = StochasticTrainingConfig(),
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
    if objective not in {"imitation", "regret", "stochastic"}:
        raise ValueError("objective must be 'imitation', 'regret', or 'stochastic'")
    if objective == "stochastic" and model_kind != "transformer":
        raise ValueError("stochastic geometry training requires model_kind='transformer'")
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
        elif objective == "regret":

            def task_loss(p: Params, k: Any) -> Any:
                return unrolled_regret_loss(p, problem, k, config, unroll, model_kind)

        else:

            def task_loss(p: Params, k: Any) -> Any:
                return stochastic_geometry_loss(p, problem, k, config, stochastic)

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
