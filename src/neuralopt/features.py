"""Dimensionless, bounded trajectory features."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def normalize_coordinates(x: Any, lo: Any, hi: Any, eps: float = 1e-12) -> Any:
    """Map a finite box to ``[-1, 1]`` without dividing by a tiny width."""
    width = jnp.maximum(hi - lo, eps)
    return 2.0 * (x - lo) / width - 1.0


def denormalize_coordinates(u: Any, lo: Any, hi: Any) -> Any:
    """Inverse of :func:`normalize_coordinates`."""
    return lo + 0.5 * (u + 1.0) * (hi - lo)


def gradient_to_normalized(grad_x: Any, lo: Any, hi: Any) -> Any:
    """Transform ``df/dx`` to ``df/du`` for ``u in [-1, 1]``."""
    return 0.5 * grad_x * (hi - lo)


def parameter_features(x: Any, grad: Any, lo: Any, hi: Any, best_x: Any) -> Any:
    """Return one token per coordinate with translation/scale normalization."""
    xn = normalize_coordinates(x, lo, hi)
    bn = normalize_coordinates(best_x, lo, hi)
    gs = gradient_to_normalized(grad, lo, hi)
    gn = gs / (jnp.linalg.norm(gs) + 1e-8)
    return jnp.stack((xn, bn, gn, jnp.abs(gn), jnp.ones_like(xn)), axis=-1)
