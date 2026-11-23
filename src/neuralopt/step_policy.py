"""v0.8-B learned step policy (research mode): per-coordinate log step sizes from a
transformer over the recent trajectory. Plan: ``docs/plans/neuralopt-v08b-step-policy.md``
§3.1. Nothing here is trained; training is pre-registered separately.

All inputs are in box-normalized coordinates ``u = 2 (x − lo)/(hi − lo) − 1`` with
gradients ``g_u = Jᵀ g_x``, ``J = diag((hi − lo)/2)``. The history window holds the last
``H`` evaluated states (position, gradient, value); unfilled slots are masked.

Per coordinate j and history slot τ the token is
``(u, g̃, Δu, Δg̃, value rank, is-best, budget fraction, Δ-valid)``, with ``g̃`` the gradient
divided by the window's RMS gradient. A small transformer attends over time for each
coordinate (shared weights), one attention layer mixes the coordinate tokens (no positional
encoding, so the policy is equivariant to coordinate permutations), and a head outputs the
mean and log standard deviation of a log step size ``η_j``. The update is
``u ← clip(u − exp(η) ⊙ g̃, −1, 1)``. Every new point costs one objective and one gradient
call (2 units).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .start_predictor import StartConfig, encode, init_start_model, observation_tokens

POLICY_SCHEMA = "step-policy-v1"
TOKEN_FEATURES = 8


@dataclass(frozen=True)
class PolicyConfig:
    history: int = 16
    width: int = 32
    heads: int = 4
    layers: int = 2
    mlp_width: int = 64
    head_width: int = 32
    log_step_bound: float = 6.0
    sigma_min: float = 0.05
    sigma_max: float = 2.0
    context: bool = False  # Family 1 only: an instance-context set encoder
    context_width: int = 32


class Window(NamedTuple):
    u: Any  # (H, d)
    g: Any  # (H, d) gradient in u coordinates
    f: Any  # (H,)
    mask: Any  # (H,) bool; newest state last


def empty_window(d: int, config: PolicyConfig) -> Window:
    h = config.history
    return Window(jnp.zeros((h, d)), jnp.zeros((h, d)), jnp.zeros(h), jnp.zeros(h, bool))


def push_window(w: Window, u: Any, g: Any, f: Any) -> Window:
    return Window(jnp.concatenate([w.u[1:], u[None]]), jnp.concatenate([w.g[1:], g[None]]),
                  jnp.concatenate([w.f[1:], jnp.asarray(f, w.f.dtype)[None]]),
                  jnp.concatenate([w.mask[1:], jnp.ones(1, bool)]))


# ---- parameters ------------------------------------------------------------------------


def _context_config(config: PolicyConfig) -> StartConfig:
    return StartConfig(width=config.context_width, heads=4, layers=1, mlp_width=64)


def init_policy(key: Any, config: PolicyConfig) -> dict[str, Any]:
    keys = iter(jax.random.split(key, 32))
    w = config.width

    def weight(a: int, b: int, scale: float = 1.0) -> Any:
        return jax.random.normal(next(keys), (a, b)) * scale * a**-0.5

    params: dict[str, Any] = {"embed": weight(TOKEN_FEATURES, w), "embed_b": jnp.zeros(w)}
    for i in [*range(config.layers), "mix"]:
        for name in ("q", "k", "v", "o"):
            params[f"{name}{i}"] = weight(w, w)
        params[f"mlp_in{i}"] = weight(w, config.mlp_width)
        params[f"mlp_in_b{i}"] = jnp.zeros(config.mlp_width)
        params[f"mlp_out{i}"] = weight(config.mlp_width, w)
        params[f"mlp_out_b{i}"] = jnp.zeros(w)
    params["head_in"] = weight(w, config.head_width)
    params["head_in_b"] = jnp.zeros(config.head_width)
    params["head_out"] = weight(config.head_width, 2, 0.1)
    params["head_out_b"] = jnp.zeros(2)
    if config.context:
        encoder = init_start_model(next(keys), _context_config(config))
        for k, v in encoder.items():
            if not k.startswith(("cand_", "conf_")):
                params[f"ctx_{k}"] = v
        params["ctx_proj"] = weight(2 * config.context_width, w)
    return params


def parameter_count(params: dict[str, Any]) -> int:
    return int(sum(np.size(v) for v in params.values()))


# ---- forward ---------------------------------------------------------------------------


def _layer_norm(z: Any) -> Any:
    mean = jnp.mean(z, axis=-1, keepdims=True)
    variance = jnp.mean((z - mean) ** 2, axis=-1, keepdims=True)
    return (z - mean) * jax.lax.rsqrt(variance + 1e-5)


def _block(z: Any, params: dict[str, Any], i: Any, heads: int, valid: Any) -> Any:
    n, width = z.shape
    depth = width // heads

    def split(a: Any) -> Any:
        return a.reshape(n, heads, depth).transpose(1, 0, 2)

    x = _layer_norm(z)
    q, k, v = (split(x @ params[f"{name}{i}"]) for name in ("q", "k", "v"))
    logits = jnp.einsum("hnd,hmd->hnm", q, k) / jnp.sqrt(depth)
    logits = jnp.where(valid[None, None, :], logits, -1e30)
    att = jnp.einsum("hnm,hmd->hnd", jax.nn.softmax(logits, axis=-1), v)
    z = z + att.transpose(1, 0, 2).reshape(n, width) @ params[f"o{i}"]
    h = jax.nn.gelu(_layer_norm(z) @ params[f"mlp_in{i}"] + params[f"mlp_in_b{i}"])
    return z + h @ params[f"mlp_out{i}"] + params[f"mlp_out_b{i}"]


def _tokens(w: Window, budget_fraction: Any) -> Any:
    """(d, H, 8) tokens; masked slots are zero."""
    m = w.mask
    mf = m.astype(w.u.dtype)
    rms = jnp.sqrt(jnp.sum((w.g * w.g) * mf[:, None]) / jnp.maximum(jnp.sum(mf) * w.g.shape[1],
                                                                     1.0)) + 1e-30
    g = w.g / rms
    prev_valid = jnp.concatenate([jnp.zeros(1, bool), m[:-1]]) & m
    du = jnp.where(prev_valid[:, None], w.u - jnp.roll(w.u, 1, axis=0), 0.0)
    dg = jnp.where(prev_valid[:, None], g - jnp.roll(g, 1, axis=0), 0.0)
    f = jnp.where(m, w.f, jnp.inf)
    best = jnp.min(f)
    worst = jnp.max(jnp.where(m, w.f, -jnp.inf))
    rank = jnp.where(m, (w.f - best) / (worst - best + 1e-30), 0.0)
    is_best = (m & (w.f == best)).astype(w.u.dtype)
    hh, d = w.u.shape
    scal = jnp.stack([rank, is_best, jnp.full(hh, budget_fraction, w.u.dtype),
                      prev_valid.astype(w.u.dtype)], axis=-1)  # (H, 4)
    per = jnp.stack([w.u, g, du, dg], axis=-1)  # (H, d, 4)
    tokens = jnp.concatenate([per, jnp.broadcast_to(scal[:, None, :], (hh, d, 4))], axis=-1)
    tokens = tokens * mf[:, None, None]
    return jnp.transpose(tokens, (1, 0, 2))


def policy_outputs(params: dict[str, Any], w: Window, budget_fraction: Any,
                   config: PolicyConfig, context: Any = None,
                   context_stats: Any = None) -> tuple[Any, Any]:
    """Per-coordinate mean and log standard deviation of the log step size."""
    tokens = _tokens(w, budget_fraction)

    def per_coordinate(tok: Any) -> Any:
        z = tok @ params["embed"] + params["embed_b"]
        for i in range(config.layers):
            z = _block(z, params, i, config.heads, w.mask)
        return _layer_norm(z)[-1]

    z = jax.vmap(per_coordinate)(tokens)  # (d, width)
    if config.context:
        enc = {k.removeprefix("ctx_"): v for k, v in params.items()
               if k.startswith("ctx_") and k != "ctx_proj"}
        pooled = encode(enc, observation_tokens(context, context_stats), context["mask"],
                        _context_config(config))
        z = z + pooled @ params["ctx_proj"]
    z = _layer_norm(_block(z, params, "mix", config.heads, jnp.ones(z.shape[0], bool)))
    h = jax.nn.gelu(z @ params["head_in"] + params["head_in_b"])
    out = h @ params["head_out"] + params["head_out_b"]
    mu = config.log_step_bound * jnp.tanh(out[:, 0])
    lo, hi = np.log(config.sigma_min), np.log(config.sigma_max)
    log_sigma = lo + (hi - lo) * jax.nn.sigmoid(out[:, 1])
    return mu, log_sigma


# ---- rollout ---------------------------------------------------------------------------


def policy_rollout(params: dict[str, Any], objective: Any, context: Any, x0: Any, lo: Any,
                   hi: Any, budget: int, key: Any, config: PolicyConfig,
                   context_stats: Any = None) -> dict[str, Any]:
    """Run the stochastic policy from ``x0`` for ``budget // 2`` points (2 units each)."""
    lo, hi = jnp.asarray(lo), jnp.asarray(hi)
    points = budget // 2
    vg = jax.value_and_grad(lambda x: objective(x, context))
    half = (hi - lo) / 2.0
    d = lo.shape[0]

    def step(carry: Any, k: Any) -> tuple[Any, Any]:
        x, w, key = carry
        f, g = vg(x)
        ok = jnp.isfinite(f) & jnp.all(jnp.isfinite(g))
        g_u = jnp.where(ok, g * half, 0.0)
        u = (x - lo) / half - 1.0
        w = push_window(w, u, g_u, jnp.where(ok, f, jnp.inf))
        used = 2 * (k + 1)
        mu, log_sigma = policy_outputs(params, w, 1.0 - used / budget, config, context,
                                       context_stats)
        key, sub = jax.random.split(key)
        eta = mu + jnp.exp(log_sigma) * jax.random.normal(sub, mu.shape)
        mf = w.mask.astype(g_u.dtype)
        rms = jnp.sqrt(jnp.sum(w.g * w.g * mf[:, None]) / jnp.maximum(jnp.sum(mf) * d, 1.0))
        u_new = jnp.clip(u - jnp.exp(eta) * g_u / (rms + 1e-30), -1.0, 1.0)
        return (lo + (u_new + 1.0) * half, w, key), (f, x)

    _, (values, xs) = jax.lax.scan(step, (jnp.asarray(x0), empty_window(d, config), key),
                                   jnp.arange(points))
    return {"values": values, "points": xs, "units": 2 * jnp.arange(1, points + 1),
            "objective_calls": points, "gradient_calls": points}


# ---- checkpoints (fail closed) ---------------------------------------------------------


def save_policy_checkpoint(path: str | Path, params: dict[str, Any], config: PolicyConfig,
                           metadata: dict[str, Any] | None = None) -> None:
    arrays = {f"param__{k}": np.asarray(v) for k, v in params.items()}
    arrays["schema"] = np.array([POLICY_SCHEMA])
    arrays["policy_config"] = np.array([json.dumps(asdict(config), sort_keys=True)])
    arrays["metadata"] = np.array([json.dumps(metadata or {}, sort_keys=True, default=str)])
    saver: Any = np.savez_compressed
    saver(path, **arrays)


def load_policy_checkpoint(path: str | Path) -> tuple[dict[str, Any], PolicyConfig]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"][0]) if "schema" in data.files else None
        if schema != POLICY_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema for the step policy: {schema}")
        config = PolicyConfig(**json.loads(str(data["policy_config"][0])))
        params = {k.removeprefix("param__"): jnp.asarray(data[k]) for k in data.files
                  if k.startswith("param__")}
    expected = init_policy(jax.random.PRNGKey(0), config)
    if {k: v.shape for k, v in params.items()} != {k: v.shape for k, v in expected.items()}:
        raise ValueError("policy checkpoint parameters do not match its schema configuration")
    return params, config
