"""JAX-native overdamped and underdamped Langevin transitions.

Bounded problems evolve in unconstrained logistic coordinates. For temperature
``T``, the transformed potential is ``f(x(z)) - T log|dx/dz|`` so that the
continuous-time invariant density maps to the bounded Gibbs density. The Euler
and BAOAB discretizations have finite-step bias and are not exact samplers.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .hamiltonian import (
    from_unconstrained,
    log_abs_det_jacobian,
    mass_solve,
    sample_momentum,
    to_unconstrained,
    validate_metric,
)


@dataclass(frozen=True)
class OverdampedConfig:
    step_size: float = 1e-2
    temperature: float = 1.0
    steps: int = 1
    transform_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.step_size <= 0 or self.temperature <= 0 or self.steps < 1:
            raise ValueError("step_size, temperature, and steps must be positive")
        if not 0 < self.transform_epsilon < 0.5:
            raise ValueError("transform_epsilon must be in (0, 0.5)")


@dataclass(frozen=True)
class UnderdampedConfig:
    step_size: float = 1e-2
    temperature: float = 1.0
    friction: float = 1.0
    steps: int = 1
    transform_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.step_size <= 0 or self.temperature <= 0 or self.friction <= 0 or self.steps < 1:
            raise ValueError("step_size, temperature, friction, and steps must be positive")
        if not 0 < self.transform_epsilon < 0.5:
            raise ValueError("transform_epsilon must be in (0, 0.5)")


DEFAULT_OVERDAMPED_CONFIG = OverdampedConfig()
DEFAULT_UNDERDAMPED_CONFIG = UnderdampedConfig()


class LangevinTransition(NamedTuple):
    state: Any
    unconstrained_state: Any
    momentum: Any | None
    candidate_value: Any
    distance: Any
    finite: Any
    objective_evaluations: int
    gradient_evaluations: int
    integration_steps: int


def mobility_apply(vector: Any, diag: Any, lowrank: Any) -> Any:
    return diag * vector + lowrank @ (lowrank.T @ vector)


def sample_mobility_noise(key: Any, diag: Any, lowrank: Any) -> Any:
    """Sample ``N(0, diag(diag) + U U.T)``."""
    return sample_momentum(key, diag, lowrank)


def transformed_potential(objective: Any, lo: Any, hi: Any, temperature: float) -> Any:
    def potential(z: Any) -> Any:
        x = from_unconstrained(z, lo, hi)
        return objective(x) - temperature * log_abs_det_jacobian(z, lo, hi)

    return potential


def _overdamped_episode(
    key: Any,
    z: Any,
    objective: Any,
    lo: Any,
    hi: Any,
    mobility_diag: Any,
    mobility_lowrank: Any,
    config: OverdampedConfig,
) -> tuple[Any, Any]:
    potential = transformed_potential(objective, lo, hi, config.temperature)
    keys = jax.random.split(key, config.steps)

    def step(position: Any, noise_key: Any) -> tuple[Any, None]:
        gradient = jax.grad(potential)(position)
        noise = sample_mobility_noise(noise_key, mobility_diag, mobility_lowrank)
        next_position = (
            position
            - config.step_size * mobility_apply(gradient, mobility_diag, mobility_lowrank)
            + jnp.sqrt(2.0 * config.temperature * config.step_size) * noise
        )
        return next_position, None

    next_z, _ = jax.lax.scan(step, z, keys)
    state = from_unconstrained(next_z, lo, hi)
    return next_z, objective(state)


_compiled_overdamped = jax.jit(_overdamped_episode, static_argnames=("objective", "config"))


def overdamped_transition(
    key: Any,
    state: Any,
    objective: Any,
    lo: Any,
    hi: Any,
    mobility_diag: Any,
    mobility_lowrank: Any,
    config: OverdampedConfig = DEFAULT_OVERDAMPED_CONFIG,
    unconstrained_state: Any | None = None,
) -> LangevinTransition:
    """Run an Euler–Maruyama episode with episode-frozen mobility."""
    validate_metric(mobility_diag, mobility_lowrank)
    z = (
        to_unconstrained(state, lo, hi, config.transform_epsilon)
        if unconstrained_state is None
        else unconstrained_state
    )
    next_z, value = _compiled_overdamped(
        key, z, objective, lo, hi, mobility_diag, mobility_lowrank, config
    )
    next_state = from_unconstrained(next_z, lo, hi)
    return LangevinTransition(
        next_state,
        next_z,
        None,
        value,
        jnp.linalg.norm(next_state - state),
        jnp.all(jnp.isfinite(next_z))
        & jnp.all(jnp.isfinite(next_state))
        & jnp.isfinite(value),
        config.steps + 1,
        config.steps,
        config.steps,
    )


def _underdamped_episode(
    key: Any,
    z: Any,
    momentum: Any,
    objective: Any,
    lo: Any,
    hi: Any,
    mass_diag: Any,
    mass_lowrank: Any,
    config: UnderdampedConfig,
) -> tuple[Any, Any, Any]:
    potential = transformed_potential(objective, lo, hi, config.temperature)
    keys = jax.random.split(key, config.steps)
    damping = jnp.exp(-config.friction * config.step_size)
    noise_scale = jnp.sqrt(config.temperature * (1.0 - damping**2))

    def step(carry: tuple[Any, Any], noise_key: Any) -> tuple[tuple[Any, Any], None]:
        position, p = carry
        p = p - 0.5 * config.step_size * jax.grad(potential)(position)
        position = position + 0.5 * config.step_size * mass_solve(p, mass_diag, mass_lowrank)
        p = damping * p + noise_scale * sample_momentum(noise_key, mass_diag, mass_lowrank)
        position = position + 0.5 * config.step_size * mass_solve(p, mass_diag, mass_lowrank)
        p = p - 0.5 * config.step_size * jax.grad(potential)(position)
        return (position, p), None

    (next_z, next_momentum), _ = jax.lax.scan(step, (z, momentum), keys)
    state = from_unconstrained(next_z, lo, hi)
    return next_z, next_momentum, objective(state)


_compiled_underdamped = jax.jit(_underdamped_episode, static_argnames=("objective", "config"))


def underdamped_transition(
    key: Any,
    state: Any,
    objective: Any,
    lo: Any,
    hi: Any,
    mass_diag: Any,
    mass_lowrank: Any,
    config: UnderdampedConfig = DEFAULT_UNDERDAMPED_CONFIG,
    unconstrained_state: Any | None = None,
    momentum: Any | None = None,
) -> LangevinTransition:
    """Run a fixed-mass BAOAB episode with correct OU momentum noise."""
    validate_metric(mass_diag, mass_lowrank)
    z = (
        to_unconstrained(state, lo, hi, config.transform_epsilon)
        if unconstrained_state is None
        else unconstrained_state
    )
    momentum_key, episode_key = jax.random.split(key)
    p = (
        jnp.sqrt(config.temperature) * sample_momentum(momentum_key, mass_diag, mass_lowrank)
        if momentum is None
        else momentum
    )
    next_z, next_momentum, value = _compiled_underdamped(
        episode_key, z, p, objective, lo, hi, mass_diag, mass_lowrank, config
    )
    next_state = from_unconstrained(next_z, lo, hi)
    return LangevinTransition(
        next_state,
        next_z,
        next_momentum,
        value,
        jnp.linalg.norm(next_state - state),
        jnp.all(jnp.isfinite(next_z))
        & jnp.all(jnp.isfinite(next_state))
        & jnp.all(jnp.isfinite(next_momentum))
        & jnp.isfinite(value),
        2 * config.steps + 1,
        2 * config.steps,
        config.steps,
    )


def diffrax_overdamped_reference(
    key: Any,
    initial_state: Any,
    objective: Any,
    step_size: float,
    steps: int,
    temperature: float,
) -> Any:
    """Optional unbounded identity-mobility Euler reference using Diffrax."""
    try:
        diffrax = importlib.import_module("diffrax")
    except ImportError as exc:
        raise ImportError("install neuralopt[sde] for Diffrax references") from exc
    if step_size <= 0 or steps < 1 or temperature <= 0:
        raise ValueError("step_size, steps, and temperature must be positive")

    def drift(t: Any, y: Any, args: Any) -> Any:
        del t, args
        return -jax.grad(objective)(y)

    def diffusion(t: Any, y: Any, args: Any) -> Any:
        del t, args
        return jnp.sqrt(2.0 * temperature) * jnp.eye(y.shape[0], dtype=y.dtype)

    end = step_size * steps
    brownian = diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=end,
        tol=step_size / 8,
        shape=(initial_state.shape[0],),
        key=key,
    )
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian)
    )
    solution = diffrax.diffeqsolve(
        terms,
        diffrax.Euler(),
        t0=0.0,
        t1=end,
        dt0=step_size,
        y0=initial_state,
        saveat=diffrax.SaveAt(t1=True),
    )
    if solution.ys is None:
        raise RuntimeError("Diffrax did not return the requested terminal state")
    return solution.ys[-1]
