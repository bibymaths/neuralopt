"""Validated problem definition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

Array = Any


@dataclass(frozen=True)
class Problem:
    """A bounded scalar minimization problem.

    ``certified_expression`` must be an :class:`neuralopt.Expr`; arbitrary JAX
    programs deliberately cannot enter certified mode.
    """

    objective: Callable[[Array], Array]
    bounds: tuple[Array, Array]
    dimension: int
    x0: Array | None = None
    gradient: Callable[[Array], Array] | None = None
    residual: Callable[[Array], Array] | None = None
    constraints: tuple[Callable[[Array], Array], ...] = ()
    max_evaluations: int = 1000
    dtype: Any = jnp.float32
    certified_expression: Any | None = None
    identifier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        lo = np.broadcast_to(np.asarray(self.bounds[0], dtype=float), (self.dimension,)).copy()
        hi = np.broadcast_to(np.asarray(self.bounds[1], dtype=float), (self.dimension,)).copy()
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("bounds must be finite")
        if np.any(lo >= hi):
            raise ValueError("each lower bound must be strictly below its upper bound")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        x0 = (lo + hi) / 2 if self.x0 is None else np.asarray(self.x0, dtype=float)
        if x0.shape != (self.dimension,):
            raise ValueError(f"x0 must have shape ({self.dimension},)")
        if not np.all(np.isfinite(x0)) or np.any(x0 < lo) or np.any(x0 > hi):
            raise ValueError("x0 must be finite and inside bounds")
        object.__setattr__(
            self, "bounds", (jnp.asarray(lo, self.dtype), jnp.asarray(hi, self.dtype))
        )
        object.__setattr__(self, "x0", jnp.asarray(x0, self.dtype))
