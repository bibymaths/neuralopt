"""Small-dimensional exact geometry utilities for Itô Langevin research."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def mobility_divergence(mobility_fn: Any, x: Any) -> Any:
    """Return ``(div A)_i = sum_j partial_j A_ij`` using exact autodiff.

    This materializes the mobility Jacobian and is intended only for controlled
    small-dimensional experiments.
    """
    jacobian = jax.jacfwd(mobility_fn)(x)
    return jnp.einsum("ijj->i", jacobian)


def ito_gibbs_drift(objective: Any, mobility_fn: Any, x: Any, temperature: float) -> Any:
    """Drift preserving ``exp(-f/T)`` in the ideal continuous-time model."""
    mobility = mobility_fn(x)
    return -mobility @ jax.grad(objective)(x) + temperature * mobility_divergence(mobility_fn, x)
