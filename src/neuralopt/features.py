"""Dimensionless, bounded trajectory features."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def parameter_features(x: Any, grad: Any, lo: Any, hi: Any, best_x: Any) -> Any:
    """Return one token per coordinate with translation/scale normalization."""
    width = hi - lo
    xn = 2 * (x - lo) / width - 1
    bn = 2 * (best_x - lo) / width - 1
    gs = grad * width
    gn = gs / (jnp.linalg.norm(gs) + 1e-8)
    return jnp.stack((xn, bn, gn, jnp.abs(gn), jnp.ones_like(xn)), axis=-1)
