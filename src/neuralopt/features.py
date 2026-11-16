"""Dimensionless, bounded trajectory features."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def safe_l2_norm(x: Any, eps: float = 1e-8) -> Any:
    """A differentiably smooth Euclidean norm, including at ``x == 0``."""
    return jnp.sqrt(jnp.sum(jnp.square(x)) + eps**2)


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


def parameter_features(
    x: Any, grad: Any, lo: Any, hi: Any, best_x: Any, scale_aware: bool = False
) -> Any:
    """Return one token per coordinate with translation/scale normalization."""
    xn = normalize_coordinates(x, lo, hi)
    bn = normalize_coordinates(best_x, lo, hi)
    gs = gradient_to_normalized(grad, lo, hi)
    gn = gs / safe_l2_norm(gs)
    if scale_aware:
        scale = jnp.tanh(jnp.log1p(safe_l2_norm(gs)) / 5.0)
        scale_feature = jnp.full_like(xn, scale)
    else:
        scale_feature = jnp.ones_like(xn)
    return jnp.stack((xn, bn, gn, jnp.abs(gn), scale_feature), axis=-1)
