"""Bounded history of actually evaluated states and masked secant features (v0.6).

Every record is ``(u, g_u, f)`` evaluated at the *same* point, in normalized box
coordinates ``u = 2 (x - lo) / (hi - lo) - 1`` with ``g_u = J g_x`` and
``J = diag((hi - lo) / 2)``. Consecutive records give ``s = Δu`` and ``y = Δg_u``;
on a quadratic ``y = H_u s`` with ``H_u = J H_x J``. ``K`` pairs identify ``H_u`` only
on ``span(S)`` (rank at most ``min(K, d)``): two quadratics whose Hessians agree on
that span, visited from the same initial record with displacements from the minimizer
inside the span, yield identical histories, so these features never imply full-Hessian
recovery. Coordinates that never moved get diagonal curvature 0, which is not
distinguished from zero curvature in this ``history-v1`` schema.
Negative ``sᵀy`` is reported (it may be genuine negative curvature or noise), never
inverted. Rejected candidates have no gradient and therefore never enter a pair.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from .features import gradient_to_normalized, parameter_features

HISTORY_FEATURE_NAMES = (
    "diag_curvature",
    "last_s",
    "last_y",
    "newton_step",
    "negative_fraction",
    "valid_fraction",
    "mean_abs_cosine",
    "anisotropy",
    "effective_rank",
    "has_history",
    "rejections",
    "quadratic_consistency",
)
HISTORY_FEATURES = len(HISTORY_FEATURE_NAMES)

_TINY = 1e-30
_STEP_TOL = 1e-7  # max-norm displacement in u units below which a pair is degenerate
_GRADIENT_RTOL = 1e-6  # gradient change below this relative level is roundoff
_CONSISTENCY_RTOL = 1e-4  # predicted decrease must exceed float32 cancellation noise


class TrajectoryHistory(NamedTuple):
    """Fixed-shape shift buffer; index ``-1`` is the newest record."""

    u: Any  # (K + 1, d)
    g: Any  # (K + 1, d)
    f: Any  # (K + 1,)
    valid: Any  # (K + 1,) bool
    rejections: Any  # () int32, consecutive rejections since the newest record


def empty_history(pairs: int, dimension: int, dtype: Any = jnp.float32) -> TrajectoryHistory:
    """Return an all-invalid buffer able to hold ``pairs`` secant pairs."""
    if pairs < 0 or dimension < 1:
        raise ValueError("pairs must be nonnegative and dimension positive")
    n = pairs + 1
    return TrajectoryHistory(
        jnp.zeros((n, dimension), dtype),
        jnp.zeros((n, dimension), dtype),
        jnp.zeros((n,), dtype),
        jnp.zeros((n,), bool),
        jnp.zeros((), jnp.int32),
    )


def push_history(h: TrajectoryHistory, u: Any, g_u: Any, f: Any) -> TrajectoryHistory:
    """Append an evaluated record, dropping the oldest, and reset rejections."""

    def shift(buffer: Any, new: Any) -> Any:
        return jnp.concatenate((buffer[1:], jnp.asarray(new, buffer.dtype)[None]), axis=0)

    return TrajectoryHistory(
        shift(h.u, u),
        shift(h.g, g_u),
        shift(h.f, f),
        shift(h.valid, True),
        jnp.zeros_like(h.rejections),
    )


def record_rejection(h: TrajectoryHistory) -> TrajectoryHistory:
    """Count a rejected candidate; it has no gradient and is not stored."""
    return h._replace(rejections=h.rejections + 1)


def _norm(x: Any, axis: int = -1) -> Any:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + _TINY)


def secant_pairs(
    h: TrajectoryHistory, step_tol: float = _STEP_TOL, gradient_rtol: float = _GRADIENT_RTOL
) -> tuple[Any, Any, Any]:
    """Return sanitized ``(S, Y, mask)``; masked rows are zero, never fake data.

    The default thresholds are the v0.6 (float32-era) ones; the v0.7 Gram schema v2
    passes float64 thresholds explicitly.
    """
    s = h.u[1:] - h.u[:-1]
    y = h.g[1:] - h.g[:-1]
    finite = (
        jnp.all(jnp.isfinite(s), axis=-1)
        & jnp.all(jnp.isfinite(y), axis=-1)
        & jnp.isfinite(h.f[1:])
        & jnp.isfinite(h.f[:-1])
    )
    gradient_scale = jnp.maximum(_norm(h.g[1:]), _norm(h.g[:-1]))
    mask = (
        h.valid[1:]
        & h.valid[:-1]
        & finite
        & (jnp.max(jnp.abs(s), axis=-1, initial=0.0) > step_tol)
        & (_norm(y) > gradient_rtol * gradient_scale)
    )
    keep = mask[:, None]
    return jnp.where(keep, s, 0.0), jnp.where(keep, y, 0.0), mask


def history_features(h: TrajectoryHistory, g_u: Any) -> Any:
    """Return ``(d, 12)`` masked, objective-scale-invariant secant features.

    Columns follow :data:`HISTORY_FEATURE_NAMES`. Per-coordinate columns are
    permutation equivariant; the rest are broadcast global statistics. With no
    valid pair every column except ``valid_fraction``, ``has_history`` and
    ``rejections`` is exactly zero (``rejections`` counts even without pairs; for
    ``K = 0`` all twelve columns are zero).
    """
    dimension = g_u.shape[-1]
    pairs = h.u.shape[0] - 1
    dtype = h.u.dtype
    if pairs == 0:
        return jnp.zeros((dimension, HISTORY_FEATURES), dtype)
    s, y, mask = secant_pairs(h)
    m = mask.astype(dtype)
    n = jnp.sum(m)
    has = n > 0
    count = jnp.maximum(n, 1.0)
    curvature = jnp.sum(s * y, axis=-1)
    sq = jnp.sum(s * s, axis=-1)
    s_norm, y_norm = _norm(s), _norm(y)
    total_sq = jnp.sum(m * sq)
    # Self-scaling curvature; the small |y||s| term only guards sᵀy = 0 on all pairs.
    rho = (jnp.sum(m * jnp.abs(curvature)) + 1e-6 * jnp.sum(m * s_norm * y_norm)) / (
        total_sq + _TINY
    )
    rho = jnp.where(has, rho, 1.0)

    diag = jnp.sum(s * y, axis=0) / (jnp.sum(s * s, axis=0) + 1e-12 * total_sq + _TINY)
    last = jnp.argmax(jnp.where(mask, jnp.arange(pairs), -1))
    last_s = s[last] / _norm(s[last])
    last_y = y[last] / _norm(y[last])
    newton = jnp.tanh(jnp.log(jnp.maximum(_norm(g_u), _TINY) / rho) / 4.0)

    negative = jnp.sum(m * (curvature < 0)) / count
    cosine = jnp.sum(m * jnp.abs(curvature) / (s_norm * y_norm)) / count
    rayleigh = jnp.abs(curvature) / (sq + _TINY)
    positive = mask & (rayleigh > 0)
    largest = jnp.max(jnp.where(positive, rayleigh, 0.0))
    smallest = jnp.min(jnp.where(positive, rayleigh, jnp.inf))
    two = jnp.sum(positive) >= 2
    ratio = largest / jnp.where(two, smallest, 1.0)
    anisotropy = jnp.where(two, jnp.tanh(jnp.log(jnp.maximum(ratio, 1.0)) / 4.0), 0.0)
    unit = s / s_norm[:, None] * m[:, None]
    gram = unit @ unit.T
    effective_rank = jnp.where(has, n**2 / (jnp.sum(gram**2) + _TINY), 0.0) / pairs

    predicted = 0.5 * jnp.sum((h.g[1:] + h.g[:-1]) * s, axis=-1)
    df = jnp.where(mask, h.f[1:] - h.f[:-1], 0.0)
    consistent = mask & (
        jnp.abs(predicted) > _CONSISTENCY_RTOL * (jnp.abs(jnp.where(mask, h.f[1:], 0.0))
                                                  + jnp.abs(jnp.where(mask, h.f[:-1], 0.0)))
        + _TINY
    )
    safe_predicted = jnp.where(consistent, predicted, 1.0)
    deviation = jnp.clip(df / safe_predicted - 1.0, -10.0, 10.0)
    quadratic = jnp.tanh(
        jnp.sum(jnp.where(consistent, deviation, 0.0)) / jnp.maximum(jnp.sum(consistent), 1)
    )

    def coord(values: Any) -> Any:
        return jnp.where(has, values, 0.0)

    def broadcast(value: Any) -> Any:
        return jnp.full((dimension,), value, dtype)

    columns = (
        coord(jnp.arcsinh(diag / rho) / 4.0),
        coord(last_s),
        coord(last_y),
        broadcast(jnp.where(has, newton, 0.0)),
        broadcast(negative),
        broadcast(n / pairs),
        broadcast(cosine),
        broadcast(anisotropy),
        broadcast(effective_rank),
        broadcast(has.astype(dtype)),
        broadcast(jnp.tanh(h.rejections.astype(dtype) / 4.0)),
        broadcast(jnp.where(has, quadratic, 0.0)),
    )
    return jnp.stack(columns, axis=-1).astype(dtype)


def history_token_features(x: Any, grad: Any, lo: Any, hi: Any, history: TrajectoryHistory) -> Any:
    """``history-v1`` tokens: scale-aware legacy five columns, then twelve history columns."""
    legacy = parameter_features(x, grad, lo, hi, x, True)
    return jnp.concatenate(
        (legacy, history_features(history, gradient_to_normalized(grad, lo, hi))), axis=-1
    )


def preconditioned_candidate(
    u: Any, unit_gradient: Any, diag: Any, lowrank: Any, step: Any, shrink: Any
) -> Any:
    """Shared v0.6 step ``clip(u - shrink * step * (diag(d) + U Uᵀ) ĝ, -1, 1)``."""
    direction = diag * unit_gradient + lowrank @ (lowrank.T @ unit_gradient)
    return jnp.clip(u - shrink * step * direction, -1.0, 1.0)
