"""Weak-form Fokker–Planck diagnostics for low-dimensional experiments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def generator_value(test_function: Any, x: Any, drift: Any, covariance: Any) -> Any:
    """Evaluate ``L phi = b.grad(phi) + 0.5 tr(a Hessian(phi))``."""
    gradient = jax.grad(test_function)(x)
    hessian = jax.hessian(test_function)(x)
    return drift @ gradient + 0.5 * jnp.sum(covariance * hessian)


def weak_form_residual(
    samples: Any,
    next_samples: Any,
    dt: float,
    test_function: Any,
    drift_fn: Any,
    covariance_fn: Any,
) -> Any:
    """Monte Carlo residual of the one-step weak generator identity."""
    observed = jnp.mean(jax.vmap(test_function)(next_samples))
    observed -= jnp.mean(jax.vmap(test_function)(samples))
    predicted = jnp.mean(
        jax.vmap(lambda x: generator_value(test_function, x, drift_fn(x), covariance_fn(x)))(
            samples
        )
    )
    return observed / dt - predicted


def gaussian_free_energy(mean: Any, covariance: Any, quadratic: Any, temperature: float) -> Any:
    """Free energy up to constants for a Gaussian and quadratic objective."""
    expected_objective = 0.5 * (mean @ quadratic @ mean + jnp.trace(quadratic @ covariance))
    _, logdet = jnp.linalg.slogdet(covariance)
    entropy_term = -0.5 * temperature * logdet
    return expected_objective + entropy_term
