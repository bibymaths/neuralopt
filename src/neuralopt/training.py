"""CPU-capable supervised policy training and portable checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from .features import (
    denormalize_coordinates,
    gradient_to_normalized,
    normalize_coordinates,
    parameter_features,
    safe_l2_norm,
)
from .hamiltonian import from_unconstrained, log_abs_det_jacobian, mass_solve, to_unconstrained
from .langevin import mobility_apply, sample_mobility_noise
from .model import (
    DEFAULT_CONFIG,
    HISTORY_SCHEMA,
    ModelConfig,
    Params,
    apply_mlp_model,
    apply_model,
    init_mlp_model,
    init_model,
    precondition,
)
from .problem import Problem
from .trajectory import (
    empty_history,
    history_token_features,
    preconditioned_candidate,
    push_history,
    record_rejection,
)


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
    out = apply(
        params,
        parameter_features(x, grad, lo, hi, problem.x0, config.scale_aware),
        1.0,
        config,
    )
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
    geometry_structure: Literal["diagonal", "lowrank", "full"] = "full"
    gradient_samples: int = 1
    temperature_range: tuple[float, float] | None = None
    friction_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.integrator not in {"overdamped", "underdamped"}:
            raise ValueError("integrator must be 'overdamped' or 'underdamped'")
        if self.episodes < 1 or self.integration_steps < 1:
            raise ValueError("episodes and integration_steps must be positive")
        if self.geometry_structure not in {"diagonal", "lowrank", "full"}:
            raise ValueError("geometry_structure must be 'diagonal', 'lowrank', or 'full'")
        if self.gradient_samples < 1:
            raise ValueError("gradient_samples must be positive")
        if not self.horizons or any(h < 1 or h > self.episodes for h in self.horizons):
            raise ValueError("horizons must lie in [1, episodes]")
        if self.step_size <= 0 or self.temperature <= 0 or self.friction <= 0:
            raise ValueError("step_size, temperature, and friction must be positive")
        if (
            min(
                self.stability_weight,
                self.compute_weight,
                self.fp_weight,
                self.free_energy_weight,
            )
            < 0
        ):
            raise ValueError("loss weights must be nonnegative")
        if self.integrator != "overdamped" and self.fp_weight:
            raise ValueError("weak Fokker--Planck training currently supports overdamped dynamics")
        for name, interval in (
            ("temperature_range", self.temperature_range),
            ("friction_range", self.friction_range),
        ):
            if interval is not None and not (0 < interval[0] <= interval[1]):
                raise ValueError(f"{name} must contain positive ordered endpoints")


def _quadratic_euler_weak_residual(
    position: Any, drift: Any, covariance_trace: Any, step_size: float
) -> Any:
    """Conditional Euler weak residual for ``phi(z) = mean(z**2)``."""
    drifted = position + step_size * drift
    observed = jnp.mean(drifted**2) - jnp.mean(position**2)
    observed += step_size * covariance_trace / position.size
    generator = 2.0 * jnp.mean(position * drift) + covariance_trace / position.size
    return observed / step_size - generator


def _single_stochastic_geometry_loss(
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
    start_key, trajectory_key, hyper_key = jax.random.split(key, 3)
    temperature_key, friction_key = jax.random.split(hyper_key)
    temperature = jnp.asarray(stochastic.temperature)
    if stochastic.temperature_range is not None:
        low, high = stochastic.temperature_range
        temperature = jnp.exp(
            jax.random.uniform(temperature_key, (), minval=jnp.log(low), maxval=jnp.log(high))
        )
    friction = jnp.asarray(stochastic.friction)
    if stochastic.friction_range is not None:
        low, high = stochastic.friction_range
        friction = jnp.exp(
            jax.random.uniform(friction_key, (), minval=jnp.log(low), maxval=jnp.log(high))
        )
    x0 = jax.random.uniform(start_key, (problem.dimension,), minval=lo, maxval=hi)
    z0 = to_unconstrained(x0, lo, hi, stochastic.transform_epsilon)
    f0 = problem.objective(x0)
    optimum = jnp.asarray(problem.metadata["known_optimum"], dtype=f0.dtype)
    regret_scale = jnp.maximum(jnp.abs(f0 - optimum), 1.0)
    if stochastic.integrator == "underdamped":
        momentum_key, trajectory_key = jax.random.split(trajectory_key)
        initial_gradient = jax.grad(problem.objective)(x0)
        initial_geometry = apply_model(
            params,
            parameter_features(x0, initial_gradient, lo, hi, x0, config.scale_aware),
            1.0,
            config,
        )
        initial_lowrank = initial_geometry["lowrank"]
        initial_diag = initial_geometry["diag"]
        if stochastic.geometry_structure == "diagonal":
            initial_lowrank = jnp.zeros_like(initial_lowrank)
        elif stochastic.geometry_structure == "lowrank":
            initial_diag = jnp.ones_like(initial_diag)
        momentum0 = jnp.sqrt(temperature) * sample_mobility_noise(
            momentum_key, initial_diag, initial_lowrank
        )
    else:
        momentum0 = jnp.zeros_like(z0)
    episode_keys = jax.random.split(trajectory_key, stochastic.episodes)
    damping = jnp.exp(-friction * stochastic.step_size)

    def potential(z: Any) -> Any:
        x = from_unconstrained(z, lo, hi)
        return problem.objective(x) - temperature * log_abs_det_jacobian(z, lo, hi)

    def episode(
        carry: tuple[Any, Any, Any, Any], inputs: tuple[Any, Any]
    ) -> tuple[tuple[Any, Any, Any, Any], tuple[Any, ...]]:
        z, momentum, best, best_x = carry
        episode_index, episode_key = inputs
        x = from_unconstrained(z, lo, hi)
        gradient = jax.grad(problem.objective)(x)
        out = apply_model(
            params,
            parameter_features(x, gradient, lo, hi, best_x, config.scale_aware),
            1.0 - episode_index / stochastic.episodes,
            config,
        )
        diag, lowrank = out["diag"], out["lowrank"]
        if stochastic.geometry_structure == "diagonal":
            lowrank = jnp.zeros_like(lowrank)
        elif stochastic.geometry_structure == "lowrank":
            diag = jnp.ones_like(diag)
        step_keys = jax.random.split(episode_key, stochastic.integration_steps)
        fp_residual = jnp.asarray(0.0, dtype=f0.dtype)

        if stochastic.integrator == "overdamped":

            def overdamped_step(position: Any, noise_key: Any) -> tuple[Any, Any]:
                potential_gradient = jax.grad(potential)(position)
                drift = -mobility_apply(potential_gradient, diag, lowrank)
                noise = sample_mobility_noise(noise_key, diag, lowrank)
                diffusion = jnp.sqrt(2.0 * temperature * stochastic.step_size) * noise
                next_position = position + stochastic.step_size * drift + diffusion
                if stochastic.fp_weight:
                    covariance_trace = 2.0 * temperature * (jnp.sum(diag) + jnp.sum(lowrank**2))
                    # For the quadratic test function, integrate the Euler noise
                    # contribution exactly. A sampled squared noise term has nonzero
                    # variance that would otherwise reward collapsing the mobility.
                    residual = _quadratic_euler_weak_residual(
                        position, drift, covariance_trace, stochastic.step_size
                    )
                else:
                    residual = jnp.asarray(0.0, dtype=f0.dtype)
                return next_position, residual**2

            next_z, residuals = jax.lax.scan(overdamped_step, z, step_keys)
            next_momentum = momentum
            fp_residual = jnp.mean(residuals)
        else:

            def baoab_step(state: tuple[Any, Any], noise_key: Any) -> tuple[tuple[Any, Any], None]:
                position, p = state
                p = p - 0.5 * stochastic.step_size * jax.grad(potential)(position)
                position = position + 0.5 * stochastic.step_size * mass_solve(p, diag, lowrank)
                noise_scale = jnp.sqrt(temperature * (1.0 - damping**2))
                p = damping * p + noise_scale * sample_mobility_noise(noise_key, diag, lowrank)
                position = position + 0.5 * stochastic.step_size * mass_solve(p, diag, lowrank)
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
        free_energy_surrogate = jnp.mean(values) - 0.5 * temperature * logdet
    else:
        free_energy_surrogate = jnp.asarray(0.0, dtype=f0.dtype)
    return (
        regret_loss
        + stochastic.stability_weight * jnp.mean(displacements)
        + stochastic.compute_weight * compute_loss
        + stochastic.fp_weight * jnp.mean(jnp.log1p(fp_residuals))
        + stochastic.free_energy_weight * free_energy_surrogate
    )


def stochastic_geometry_loss(
    params: Params,
    problem: Problem,
    key: Any,
    config: ModelConfig,
    stochastic: StochasticTrainingConfig = StochasticTrainingConfig(),
) -> Any:
    """Average pathwise stochastic-geometry losses over explicit trajectory keys."""
    if stochastic.gradient_samples == 1:
        return _single_stochastic_geometry_loss(params, problem, key, config, stochastic)
    keys = jax.random.split(key, stochastic.gradient_samples)
    losses = jax.vmap(
        lambda sample_key: _single_stochastic_geometry_loss(
            params, problem, sample_key, config, stochastic
        )
    )(keys)
    return jnp.mean(losses)


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
            parameter_features(x, grad, lo, hi, x, config.scale_aware),
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


@dataclass(frozen=True)
class HistoryUnrollConfig:
    """Differentiable mirror of the v0.6 ``preconditioned`` inference loop."""

    steps: int = 24
    horizons: tuple[int, ...] = (6, 12, 24)
    candidate_weight: float = 0.1
    regret_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.steps < 1 or not self.horizons:
            raise ValueError("steps and horizons must be positive")
        if any(h < 1 or h > self.steps for h in self.horizons):
            raise ValueError("horizons must lie in [1, steps]")
        if self.candidate_weight < 0 or self.regret_floor <= 0:
            raise ValueError("candidate_weight must be nonnegative and regret_floor positive")


def history_unroll(
    params: Params,
    problem: Problem,
    key: Any,
    config: ModelConfig,
    unroll: HistoryUnrollConfig = HistoryUnrollConfig(),
    model_kind: str = "transformer",
    x0: Any | None = None,
) -> dict[str, Any]:
    """Run the shared loop rule for ``unroll.steps`` candidates under ``lax.scan``.

    Acceptance is hard (``f' < f`` with finite value and gradient), exactly as in
    inference; the budget fraction uses the same ``1 - used / budget`` definition with
    ``budget = 2 + 2 * steps``. The candidate gradient is computed for every step here
    only because ``scan`` needs static shapes; inference charges it only on acceptance.
    History *features* are ``stop_gradient``-ed, but the meta-gradient still flows
    through positions and through ``∇f`` at earlier candidates (the normalized gradient
    direction), i.e. it uses second-order derivatives of the objective in training.
    """
    lo, hi = problem.bounds
    budget = 2 + 2 * unroll.steps
    if x0 is None:
        x0 = jax.random.uniform(key, (problem.dimension,), minval=lo, maxval=hi)
    grad_fn = jax.grad(problem.objective)
    f0 = problem.objective(x0)
    g0 = grad_fn(x0)
    history_on = config.feature_set != "legacy"
    history = empty_history(config.history_length, problem.dimension, f0.dtype)
    if history_on:
        history = push_history(
            history, normalize_coordinates(x0, lo, hi), gradient_to_normalized(g0, lo, hi), f0
        )
    apply = apply_model if model_kind == "transformer" else apply_mlp_model

    def step(carry: tuple[Any, ...], _: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        x, value, grad, hist, shrink, used = carry
        g_u = gradient_to_normalized(grad, lo, hi)
        unit = g_u / safe_l2_norm(g_u)
        if history_on:
            tokens = history_token_features(x, grad, lo, hi, hist)
        else:
            tokens = parameter_features(x, grad, lo, hi, x, config.scale_aware)
        out = apply(params, jax.lax.stop_gradient(tokens), 1.0 - used / budget, config)
        u_new = preconditioned_candidate(
            normalize_coordinates(x, lo, hi), unit, out["diag"], out["lowrank"],
            0.5 * out["step"], shrink,
        )
        candidate = jnp.clip(denormalize_coordinates(u_new, lo, hi), lo, hi)
        candidate_value = problem.objective(candidate)
        candidate_grad = grad_fn(candidate)
        accepted = (
            jnp.isfinite(candidate_value)
            & (candidate_value < value)
            & jnp.all(jnp.isfinite(candidate_grad))
        )
        pushed = push_history(
            hist,
            normalize_coordinates(candidate, lo, hi),
            gradient_to_normalized(candidate_grad, lo, hi),
            candidate_value,
        )
        next_hist = jax.tree_util.tree_map(
            lambda a, b: jnp.where(accepted, a, b), pushed, record_rejection(hist)
        )
        next_carry = (
            jnp.where(accepted, candidate, x),
            jnp.where(accepted, candidate_value, value),
            jnp.where(accepted, candidate_grad, grad),
            next_hist,
            jnp.where(accepted, 1.0, 0.5 * shrink),
            used + 1 + accepted.astype(used.dtype),
        )
        return next_carry, {
            "values": next_carry[1],
            "candidate_values": jnp.where(jnp.isfinite(candidate_value), candidate_value, value),
            "accepted": accepted,
        }

    init = (x0, f0, g0, history, jnp.asarray(1.0, f0.dtype), jnp.asarray(2.0, f0.dtype))
    final, outputs = jax.lax.scan(step, init, None, length=unroll.steps)
    outputs["initial_value"] = f0
    outputs["final_history"] = final[3]
    return outputs


def history_regret_loss(
    params: Params,
    problem: Problem,
    key: Any,
    config: ModelConfig,
    unroll: HistoryUnrollConfig = HistoryUnrollConfig(),
    model_kind: str = "transformer",
) -> Any:
    """Multi-horizon log incumbent regret plus a weighted anytime candidate term.

    Regret is normalized by the initial gap, so the loss is objective-scale invariant.
    """
    if "known_optimum" not in problem.metadata:
        raise ValueError("history training requires problem.metadata['known_optimum']")
    rolled = history_unroll(params, problem, key, config, unroll, model_kind)
    f0 = rolled["initial_value"]
    optimum = jnp.asarray(problem.metadata["known_optimum"], dtype=f0.dtype)
    scale = jnp.abs(f0 - optimum) + 1e-12

    def log_regret(values: Any) -> Any:
        return jnp.log(jnp.maximum(values - optimum, 0.0) / scale + unroll.regret_floor)

    horizon_indices = jnp.asarray([h - 1 for h in unroll.horizons])
    incumbent = jnp.mean(log_regret(rolled["values"][horizon_indices]))
    candidates = jnp.mean(log_regret(rolled["candidate_values"]))
    return incumbent + unroll.candidate_weight * candidates


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
    history_unroll: HistoryUnrollConfig = HistoryUnrollConfig(),
) -> tuple[Params, list[float]]:
    """Train with Optax; objective optimization inference never uses Optax."""
    try:
        import optax
    except ImportError as exc:
        raise ImportError("install neuralopt[train] to train") from exc
    items = list(problems)
    if not items:
        raise ValueError("at least one training problem is required")
    if objective not in {"imitation", "regret", "stochastic", "history"}:
        raise ValueError("objective must be 'imitation', 'regret', 'stochastic', or 'history'")
    if config.feature_set != "legacy" and objective != "history":
        raise ValueError("history feature sets require objective='history'")
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

        elif objective == "history":

            def task_loss(p: Params, k: Any) -> Any:
                return history_regret_loss(p, problem, k, config, history_unroll, model_kind)

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
    values = [
        config.width,
        config.heads,
        config.rank,
        config.layers,
        float(config.scale_aware),
        config.geometry_bound,
        config.lowrank_scale,
    ]
    if config.feature_set != "legacy":
        # Legacy files stay byte-compatible; versioned schemas are explicit and opt-in.
        values += [1.0, config.history_length]
        arrays["schema"] = np.array([HISTORY_SCHEMA])
    arrays.update({"config": np.array(values), "metadata": np.array([repr(metadata or {})])})
    saver: Any = np.savez_compressed
    saver(path, **arrays)


def load_checkpoint(path: str | Path) -> tuple[Params, ModelConfig]:
    with np.load(path, allow_pickle=False) as data:
        c = data["config"].tolist()
        schema = str(data["schema"][0]) if "schema" in data.files else None
        if schema is None and len(c) > 7:
            raise ValueError("extended checkpoint config without a schema tag")
        if schema is not None and (schema != HISTORY_SCHEMA or len(c) != 9 or c[7] != 1.0):
            raise ValueError(f"unsupported checkpoint schema: {schema}")
        if schema is not None and (c[8] != int(c[8]) or c[8] < 0):
            raise ValueError("checkpoint schema has an invalid history_length")
        history = (
            {"feature_set": "history-v1", "history_length": int(c[8])} if schema else {}
        )
        config = ModelConfig(
            width=int(c[0]),
            heads=int(c[1]),
            rank=int(c[2]),
            layers=int(c[3]),
            scale_aware=bool(c[4]) if len(c) > 4 else False,
            geometry_bound=float(c[5]) if len(c) > 5 else 0.0,
            lowrank_scale=float(c[6]) if len(c) > 6 else 1.0,
            **history,
        )
        params = {
            k.removeprefix("param__"): jnp.asarray(data[k])
            for k in data.files
            if k.startswith("param__")
        }
    # Fail closed when the input width contradicts the (possibly absent) schema tag.
    for key, extra in (("embed", 0), ("hidden", 2)):
        if key in params and params[key].shape[0] != config.input_features + extra:
            raise ValueError(
                f"checkpoint {key} width {params[key].shape[0]} contradicts its schema "
                f"(feature_set={config.feature_set!r})"
            )
    return params, config
