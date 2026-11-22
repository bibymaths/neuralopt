"""Gram-space history model (v0.7 Track B, opt-in; schema ``gram-v1``).

Tokens: one per stored secant pair plus one for the current gradient (``K + 1`` tokens,
independent of the dimension ``d``). Every input is invariant to rotations of the
normalized coordinates and to positive rescaling of the objective: normalized inner
products among the columns of ``S``, ``Y`` and ``g_u``, log norm ratios, masks, pair
age, the rejection counter, the last rejected trial's step length and the ratio of its
observed objective change to the predicted first-order change ``g_uᵀ Δu``. Rejected
trials were evaluated and charged, so they may inform the model; they never form
secant pairs.

Outputs ``K x K`` matrices ``A``, ``C`` and scalars ``δ``, ``σ`` define the inverse
metric (column-normalized ``Ŝ``, ``Ŷ``; masked pairs contribute nothing)::

    P = τ [ e^δ (I − Ŝ A Ŷᵀ)(I − Ŷ Aᵀ Ŝᵀ) + Ŝ C Cᵀ Ŝᵀ ] + ε τ I

applied matrix-free in ``O(K d)``. ``τ = sᵀs / sᵀy`` of the newest valid pair when
``sᵀy > 0`` (a unit-carrying inverse-curvature scale), otherwise
``fallback_length / ‖g_u‖``. ``P`` is SPD for every ``A``, ``C`` (``ε > 0``) and acts as an
*inverse* metric: the secant convention is ``P y ≈ s``. The step is
``u' = clip(u − shrink · σ · P g_u, −1, 1)`` with the un-normalized gradient.

``K`` pairs inform ``P`` only on ``span(Ŝ, Ŷ)``; on the orthogonal complement ``P`` is
``τ (e^δ + ε) I``. Nothing here is a classical solver: ``A``, ``C``, ``δ``, ``σ`` come only
from the learned model (a BFGS oracle exists only in ``scripts/gram_reference.py``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .trajectory import TrajectoryHistory, secant_pairs

GRAM_SCHEMA_V1 = "neuralopt-policy/v0.7-gram-v1"
# v2 adds the pair-mask thresholds and the no-history switch to the config. v1 files load
# with the v1 (float32-era) thresholds, which are the field defaults.
GRAM_SCHEMA = "neuralopt-policy/v0.7-gram-v2"
HISTORY_V2_MASKS = {"mask_step_tol": 1e-12, "mask_gradient_rtol": 1e-10}
_TINY = 1e-30


@dataclass(frozen=True)
class GramConfig:
    history_length: int = 8
    width: int = 32
    heads: int = 4
    layers: int = 2
    eps_rel: float = 1e-12
    fallback_length: float = 0.1
    delta_bound: float = 5.0
    sigma_bound: float = 4.0
    coefficient_bound: float = 100.0
    scalar_only: bool = False
    mask_step_tol: float = 1e-7
    mask_gradient_rtol: float = 1e-6
    history_masked: bool = False  # no-history control: every pair is masked

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history_length must be positive for the Gram model")
        if self.width % self.heads or self.layers < 1:
            raise ValueError("width must be divisible by heads and layers positive")
        if self.eps_rel < 0 or self.fallback_length <= 0:
            raise ValueError("eps_rel must be nonnegative and fallback_length positive")
        if min(self.delta_bound, self.coefficient_bound) <= 0 or self.sigma_bound <= 1:
            raise ValueError("bounds must be positive and sigma_bound > 1")
        if self.mask_step_tol < 0 or self.mask_gradient_rtol < 0:
            raise ValueError("mask thresholds must be nonnegative")

    feature_set = "gram-v1"  # class attribute: lets shared loops treat it as history-on

    @property
    def token_features(self) -> int:
        return 4 * self.history_length + 14


class GramBasis(NamedTuple):
    S_hat: Any  # (d, K) unit columns, zero where masked
    Y_hat: Any  # (d, K)
    s_norm: Any  # (K,)
    y_norm: Any  # (K,)
    mask: Any  # (K,) bool
    curvature: Any  # (K,) sᵀy (zero where masked)


class RejectionInfo(NamedTuple):
    count: Any  # consecutive rejections at the current point
    last_length: Any  # ‖Δu‖ of the last rejected trial (u units)
    last_ratio: Any  # (f(trial) − f) / (g_uᵀ Δu), clipped to [−10, 10]
    has: Any  # bool: a rejection happened at the current point


def gram_basis(history: TrajectoryHistory, config: GramConfig | None = None) -> GramBasis:
    """Normalized secant basis; ``config`` supplies the mask thresholds and switch
    (``None``: the v1 defaults)."""
    if config is None:
        S, Y, mask = secant_pairs(history)  # rows (K, d), sanitized
    else:
        S, Y, mask = secant_pairs(history, config.mask_step_tol, config.mask_gradient_rtol)
        if config.history_masked:
            mask = jnp.zeros_like(mask)
            S, Y = jnp.zeros_like(S), jnp.zeros_like(Y)
    s_norm = jnp.sqrt(jnp.sum(S * S, axis=-1) + _TINY)
    y_norm = jnp.sqrt(jnp.sum(Y * Y, axis=-1) + _TINY)
    keep = mask[:, None]
    S_hat = jnp.where(keep, S / s_norm[:, None], 0.0).T
    Y_hat = jnp.where(keep, Y / y_norm[:, None], 0.0).T
    curvature = jnp.where(mask, jnp.sum(S * Y, axis=-1), 0.0)
    return GramBasis(S_hat, Y_hat, jnp.where(mask, s_norm, 0.0), jnp.where(mask, y_norm, 0.0),
                     mask, curvature)


def _newest(mask: Any) -> Any:
    return jnp.argmax(jnp.where(mask, jnp.arange(mask.shape[0]), -1))


def tau_scale(basis: GramBasis, g_u: Any, fallback_length: float) -> Any:
    """``sᵀs / sᵀy`` of the newest valid pair if positive, else ``fallback / ‖g_u‖``."""
    i = _newest(basis.mask)
    usable = jnp.any(basis.mask) & (basis.curvature[i] > 0)
    secant = basis.s_norm[i] ** 2 / jnp.where(usable, basis.curvature[i], 1.0)
    fallback = fallback_length / jnp.sqrt(jnp.sum(g_u * g_u) + _TINY)
    return jnp.where(usable, secant, fallback)


def _masked(basis: GramBasis, A: Any, C: Any) -> tuple[Any, Any]:
    m = basis.mask.astype(basis.S_hat.dtype)
    return A * m[:, None] * m[None, :], C * m[:, None] * m[None, :]


def apply_inverse_metric(v: Any, basis: GramBasis, A: Any, C: Any, delta: Any, tau: Any,
                         eps_rel: float) -> Any:
    """``P v`` in ``O(K d)`` without forming ``P``."""
    A, C = _masked(basis, A, C)
    S, Y = basis.S_hat, basis.Y_hat
    w = v - Y @ (A.T @ (S.T @ v))
    first = w - S @ (A @ (Y.T @ w))
    second = S @ (C @ (C.T @ (S.T @ v)))
    return tau * (jnp.exp(delta) * first + second) + eps_rel * tau * v


def dense_inverse_metric(basis: GramBasis, A: Any, C: Any, delta: Any, tau: Any,
                         eps_rel: float) -> Any:
    """Dense ``P`` (tests and small-dimensional diagnostics only)."""
    A, C = _masked(basis, A, C)
    S, Y = basis.S_hat, basis.Y_hat
    eye = jnp.eye(S.shape[0], dtype=S.dtype)
    left = eye - S @ A @ Y.T
    return tau * (jnp.exp(delta) * left @ left.T + S @ C @ C.T @ S.T) + eps_rel * tau * eye


def gram_tokens(history: TrajectoryHistory, g_u: Any, rejection: RejectionInfo,
                budget_fraction: Any, config: GramConfig) -> Any:
    """``(K + 1, 4K + 14)`` rotation- and objective-scale-invariant token features."""
    K = config.history_length
    basis = gram_basis(history, config)
    dtype = basis.S_hat.dtype
    m = basis.mask.astype(dtype)
    g_norm = jnp.sqrt(jnp.sum(g_u * g_u) + _TINY)
    g_hat = g_u / g_norm
    S, Y = basis.S_hat, basis.Y_hat
    pair_gram = jnp.concatenate(
        (S.T @ S, S.T @ Y, Y.T @ S, Y.T @ Y, (S.T @ g_hat)[:, None], (Y.T @ g_hat)[:, None]),
        axis=1,
    )
    newest = _newest(basis.mask)
    any_valid = jnp.any(basis.mask)

    def log_ratio(norms: Any) -> Any:
        ref = jnp.where(any_valid, norms[newest], 1.0)
        ratio = jnp.log(jnp.where(basis.mask, norms, 1.0) / jnp.where(ref > 0, ref, 1.0))
        return jnp.where(basis.mask, jnp.tanh(ratio / 4.0), 0.0)

    age = (K - 1 - jnp.arange(K, dtype=dtype)) / K
    pair_tokens = jnp.concatenate(
        (
            pair_gram,
            log_ratio(basis.s_norm)[:, None],
            log_ratio(basis.y_norm)[:, None],
            m[:, None],
            (age * m)[:, None],
            jnp.ones((K, 1), dtype),  # is-pair
            jnp.zeros((K, 7), dtype),
        ),
        axis=1,
    )
    tau = tau_scale(basis, g_u, config.fallback_length)
    usable = any_valid & (basis.curvature[newest] > 0)
    gradient_token = jnp.concatenate(
        (
            S.T @ g_hat,
            Y.T @ g_hat,
            jnp.zeros((2 * K + 2,), dtype),
            jnp.zeros((5,), dtype),  # pair-only slots and is-pair = 0
            jnp.stack(
                (
                    usable.astype(dtype),
                    jnp.tanh(jnp.log(tau * g_norm) / 4.0),
                    jnp.tanh(jnp.asarray(rejection.count, dtype) / 4.0),
                    jnp.where(rejection.has,
                              jnp.tanh(jnp.log(jnp.asarray(rejection.last_length, dtype)
                                               + 1e-12) / 4.0), 0.0),
                    jnp.where(rejection.has, jnp.asarray(rejection.last_ratio, dtype) / 10.0,
                              0.0),
                    jnp.asarray(rejection.has, dtype),
                    jnp.asarray(budget_fraction, dtype),
                )
            ),
        )
    )
    return jnp.concatenate((pair_tokens, gradient_token[None, :]), axis=0)


def init_gram_model(key: Any, config: GramConfig) -> dict[str, Any]:
    keys = iter(jax.random.split(key, 8 + 4 * config.layers))
    w = config.width

    def weight(a: int, b: int, scale: float = 1.0) -> Any:
        return jax.random.normal(next(keys), (a, b)) * scale * a**-0.5

    params = {"embed": weight(config.token_features, w)}
    for i in range(config.layers):
        for name in ("q", "k", "v", "o"):
            params[f"{name}{i}"] = weight(w, w)
    # Small coefficient heads: an untrained model starts near A = C = 0 (P ≈ τ e^δ I).
    for name in ("a_left", "a_right", "c_left", "c_right"):
        params[name] = weight(w, w, 0.1)
    params["delta"] = weight(w, 1, 0.1)
    params["sigma"] = weight(w, 1, 0.1)
    return params


def _masked_attention(z: Any, params: dict[str, Any], i: int, heads: int, valid: Any) -> Any:
    n, width = z.shape
    depth = width // heads

    def split(a: Any) -> Any:
        return a.reshape(n, heads, depth).transpose(1, 0, 2)

    q, k, v = (split(z @ params[f"{name}{i}"]) for name in ("q", "k", "v"))
    logits = jnp.einsum("hnd,hmd->hnm", q, k) / jnp.sqrt(depth)
    logits = jnp.where(valid[None, None, :], logits, -1e30)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("hnm,hmd->hnd", weights, v).transpose(1, 0, 2).reshape(n, width)
    return z + out @ params[f"o{i}"]


def _layer_norm(z: Any) -> Any:
    mean = jnp.mean(z, axis=-1, keepdims=True)
    variance = jnp.mean((z - mean) ** 2, axis=-1, keepdims=True)
    return (z - mean) * jax.lax.rsqrt(variance + 1e-5)


def apply_gram_model(params: dict[str, Any], tokens: Any, pair_mask: Any,
                     config: GramConfig) -> dict[str, Any]:
    """Masked attention over the ``K + 1`` tokens; returns ``A``, ``C``, ``δ``, ``σ``."""
    K = config.history_length
    valid = jnp.concatenate((pair_mask, jnp.ones((1,), bool)))
    z = tokens @ params["embed"]
    for i in range(config.layers):
        z = jax.nn.gelu(_masked_attention(_layer_norm(z), params, i, config.heads, valid))
    pairs, grad_token = z[:K], z[K]
    m = pair_mask.astype(z.dtype)
    outer = m[:, None] * m[None, :]

    def bilinear(left: str, right: str) -> Any:
        raw = (pairs @ params[left]) @ (pairs @ params[right]).T / config.width
        bound = config.coefficient_bound
        return bound * jnp.tanh(raw / bound) * outer

    zeros = jnp.zeros((K, K), z.dtype)
    A = zeros if config.scalar_only else bilinear("a_left", "a_right")
    C = zeros if config.scalar_only else bilinear("c_left", "c_right")
    delta = config.delta_bound * jnp.tanh((grad_token @ params["delta"])[0])
    sigma = jnp.exp(jnp.log(config.sigma_bound) * jnp.tanh((grad_token @ params["sigma"])[0]))
    return {"A": A, "C": C, "delta": delta, "sigma": sigma}


def gram_step(params: dict[str, Any], config: GramConfig, history: TrajectoryHistory,
              g_u: Any, rejection: RejectionInfo,
              budget_fraction: Any) -> tuple[Any, Any, dict[str, Any], Any]:
    """``P g_u`` (u units), ``σ``, the model outputs and ``τ``; the step is ``σ P g_u``."""
    basis = gram_basis(history, config)
    tau = tau_scale(basis, g_u, config.fallback_length)
    tokens = gram_tokens(history, g_u, rejection, budget_fraction, config)
    out = apply_gram_model(params, tokens, basis.mask, config)
    direction = apply_inverse_metric(g_u, basis, out["A"], out["C"], out["delta"], tau,
                                     config.eps_rel)
    return direction, out["sigma"], out, tau


def gram_direction(params: dict[str, Any], config: GramConfig, history: TrajectoryHistory,
                   g_u: Any, rejection: RejectionInfo,
                   budget_fraction: Any) -> tuple[Any, dict[str, Any], Any]:
    """``σ P g_u`` (u units), the model outputs and ``τ``."""
    direction, sigma, out, tau = gram_step(params, config, history, g_u, rejection,
                                           budget_fraction)
    return sigma * direction, out, tau


def save_gram_checkpoint(path: str | Path, params: dict[str, Any], config: GramConfig,
                         metadata: dict[str, Any] | None = None) -> None:
    arrays = {f"param__{k}": np.asarray(v) for k, v in params.items()}
    arrays["schema"] = np.array([GRAM_SCHEMA])
    arrays["gram_config"] = np.array([json.dumps(asdict(config), sort_keys=True)])
    arrays["metadata"] = np.array([repr(metadata or {})])
    saver: Any = np.savez_compressed
    saver(path, **arrays)


def load_gram_checkpoint(path: str | Path) -> tuple[dict[str, Any], GramConfig]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"][0]) if "schema" in data.files else None
        if schema not in (GRAM_SCHEMA, GRAM_SCHEMA_V1):
            raise ValueError(f"unsupported checkpoint schema for the Gram model: {schema}")
        config = GramConfig(**json.loads(str(data["gram_config"][0])))
        params = {k.removeprefix("param__"): jnp.asarray(data[k]) for k in data.files
                  if k.startswith("param__")}
    expected = init_gram_model(jax.random.PRNGKey(0), config)
    if {k: v.shape for k, v in params.items()} != {k: v.shape for k, v in expected.items()}:
        raise ValueError("Gram checkpoint parameters do not match its schema configuration")
    return params, config
