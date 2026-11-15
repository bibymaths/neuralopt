"""Native JAX Hamiltonian proposals for bounded optimization experiments.

Box constraints are handled by a logistic change of variables. The Hamiltonian
therefore evolves in unconstrained coordinates and includes the log-Jacobian of
the transform. This preserves ordinary leapfrog assumptions for fixed metrics;
the transform cannot represent exact boundary points, so optimization must keep
a separate feasible incumbent archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .features import safe_l2_norm


@dataclass(frozen=True)
class HamiltonianConfig:
    step_size: float = 0.05
    leapfrog_steps: int = 5
    inverse_temperature: float = 1.0
    transform_epsilon: float = 1e-6
    max_energy_error: float = 100.0

    def __post_init__(self) -> None:
        if self.step_size <= 0 or self.leapfrog_steps < 1 or self.inverse_temperature <= 0:
            raise ValueError("step_size, leapfrog_steps, and inverse_temperature must be positive")
        if not 0 < self.transform_epsilon < 0.5:
            raise ValueError("transform_epsilon must be in (0, 0.5)")
        if self.max_energy_error <= 0:
            raise ValueError("max_energy_error must be positive")


DEFAULT_HAMILTONIAN_CONFIG = HamiltonianConfig()


class HamiltonianProposal(NamedTuple):
    state: Any
    candidate: Any
    candidate_value: Any
    accepted: Any
    log_acceptance: Any
    energy_error: Any
    divergent: Any
    distance: Any
    objective_evaluations: int
    gradient_evaluations: int
    leapfrog_steps: int


def to_unconstrained(x: Any, lo: Any, hi: Any, epsilon: float = 1e-6) -> Any:
    unit = jnp.clip((x - lo) / (hi - lo), epsilon, 1.0 - epsilon)
    return jnp.log(unit) - jnp.log1p(-unit)


def from_unconstrained(z: Any, lo: Any, hi: Any) -> Any:
    return lo + (hi - lo) * jax.nn.sigmoid(z)


def log_abs_det_jacobian(z: Any, lo: Any, hi: Any) -> Any:
    return jnp.sum(jnp.log(hi - lo) + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z))


def validate_metric(diag: Any, lowrank: Any) -> None:
    diag_array = jnp.asarray(diag)
    lowrank_array = jnp.asarray(lowrank)
    if diag_array.ndim != 1:
        raise ValueError("mass diagonal must be one-dimensional")
    if lowrank_array.ndim != 2 or lowrank_array.shape[0] != diag_array.shape[0]:
        raise ValueError("mass low-rank factor must have shape (dimension, rank)")
    if not bool(jnp.all(jnp.isfinite(diag_array))) or not bool(
        jnp.all(jnp.isfinite(lowrank_array))
    ):
        raise ValueError("mass matrix parameters must be finite")
    if not bool(jnp.all(diag_array > 0)):
        raise ValueError("mass diagonal must be strictly positive")


def mass_solve(momentum: Any, diag: Any, lowrank: Any) -> Any:
    """Apply ``(diag(diag) + U U.T)^-1`` using Woodbury algebra."""
    inverse_diag_momentum = momentum / diag
    if lowrank.shape[1] == 0:
        return inverse_diag_momentum
    inverse_diag_u = lowrank / diag[:, None]
    middle = jnp.eye(lowrank.shape[1], dtype=momentum.dtype) + lowrank.T @ inverse_diag_u
    return inverse_diag_momentum - inverse_diag_u @ jnp.linalg.solve(
        middle, lowrank.T @ inverse_diag_momentum
    )


def sample_momentum(key: Any, diag: Any, lowrank: Any) -> Any:
    """Sample momentum with covariance ``diag(diag) + U U.T``."""
    key_diag, key_rank = jax.random.split(key)
    diagonal_sample = jnp.sqrt(diag) * jax.random.normal(key_diag, diag.shape, dtype=diag.dtype)
    rank_sample = jax.random.normal(key_rank, (lowrank.shape[1],), dtype=diag.dtype)
    return diagonal_sample + lowrank @ rank_sample


def kinetic_energy(momentum: Any, diag: Any, lowrank: Any) -> Any:
    return 0.5 * momentum @ mass_solve(momentum, diag, lowrank)


def leapfrog(
    position: Any,
    momentum: Any,
    potential: Any,
    diag: Any,
    lowrank: Any,
    step_size: float,
    steps: int,
) -> tuple[Any, Any]:
    """Apply fixed-metric leapfrog without momentum reversal."""
    gradient = jax.grad(potential)(position)

    def step(carry: tuple[Any, Any, Any], _: Any) -> tuple[tuple[Any, Any, Any], None]:
        q, p, grad = carry
        half = p - 0.5 * step_size * grad
        next_q = q + step_size * mass_solve(half, diag, lowrank)
        next_grad = jax.grad(potential)(next_q)
        next_p = half - 0.5 * step_size * next_grad
        return (next_q, next_p, next_grad), None

    (result_q, result_p, _), _ = jax.lax.scan(step, (position, momentum, gradient), None, steps)
    return result_q, result_p


def make_bounded_potential(objective: Any, lo: Any, hi: Any, inverse_temperature: float) -> Any:
    def potential(z: Any) -> Any:
        x = from_unconstrained(z, lo, hi)
        return inverse_temperature * objective(x) - log_abs_det_jacobian(z, lo, hi)

    return potential


def _transition(
    key: Any,
    state: Any,
    objective: Any,
    lo: Any,
    hi: Any,
    mass_diag: Any,
    mass_lowrank: Any,
    config: HamiltonianConfig,
) -> tuple[Any, ...]:
    momentum_key, accept_key = jax.random.split(key)
    z = to_unconstrained(state, lo, hi, config.transform_epsilon)
    momentum = sample_momentum(momentum_key, mass_diag, mass_lowrank)
    potential = make_bounded_potential(objective, lo, hi, config.inverse_temperature)
    initial_energy = potential(z) + kinetic_energy(momentum, mass_diag, mass_lowrank)
    proposed_z, proposed_momentum = leapfrog(
        z,
        momentum,
        potential,
        mass_diag,
        mass_lowrank,
        config.step_size,
        config.leapfrog_steps,
    )
    # Momentum reversal makes the deterministic proposal map an involution.
    proposed_momentum = -proposed_momentum
    proposed_energy = potential(proposed_z) + kinetic_energy(
        proposed_momentum, mass_diag, mass_lowrank
    )
    energy_error = proposed_energy - initial_energy
    finite = jnp.isfinite(initial_energy) & jnp.isfinite(proposed_energy)
    divergent = (~finite) | (jnp.abs(energy_error) > config.max_energy_error)
    log_acceptance = jnp.minimum(0.0, -energy_error)
    accepted = (~divergent) & (jnp.log(jax.random.uniform(accept_key)) < log_acceptance)
    candidate = from_unconstrained(proposed_z, lo, hi)
    next_state = jnp.where(accepted, candidate, state)
    candidate_value = objective(candidate)
    return (
        next_state,
        candidate,
        candidate_value,
        accepted,
        log_acceptance,
        energy_error,
        divergent,
        safe_l2_norm(candidate - state),
    )


_compiled_transition = jax.jit(_transition, static_argnames=("objective", "config"))


def hamiltonian_proposal(
    key: Any,
    state: Any,
    objective: Any,
    lo: Any,
    hi: Any,
    mass_diag: Any,
    mass_lowrank: Any,
    config: HamiltonianConfig = DEFAULT_HAMILTONIAN_CONFIG,
) -> HamiltonianProposal:
    """Generate one Metropolis-corrected fixed-metric HMC transition."""
    validate_metric(mass_diag, mass_lowrank)
    transition = _compiled_transition(
        key, state, objective, lo, hi, mass_diag, mass_lowrank, config
    )
    return HamiltonianProposal(
        transition[0],
        transition[1],
        transition[2],
        transition[3],
        transition[4],
        transition[5],
        transition[6],
        transition[7],
        config.leapfrog_steps + 4,
        config.leapfrog_steps + 1,
        config.leapfrog_steps,
    )
