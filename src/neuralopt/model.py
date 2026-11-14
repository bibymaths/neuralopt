"""A small variable-dimensional transformer optimization policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

Params = dict[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    width: int = 32
    heads: int = 4
    rank: int = 2
    layers: int = 2


DEFAULT_CONFIG = ModelConfig()


def init_model(key: Any, config: ModelConfig = DEFAULT_CONFIG) -> Params:
    """Initialize policy parameters deterministically from a JAX key."""
    if config.width % config.heads:
        raise ValueError("width must be divisible by heads")
    keys = iter(jax.random.split(key, 8 + 4 * config.layers))

    def w(a: int, b: int, scale: float | None = None) -> Any:
        return jax.random.normal(next(keys), (a, b)) * (scale or a**-0.5)

    p: Params = {"embed": w(5, config.width), "task": w(2, config.width)}
    for i in range(config.layers):
        p[f"q{i}"], p[f"k{i}"], p[f"v{i}"], p[f"o{i}"] = (
            w(config.width, config.width) for _ in range(4)
        )
    p.update(
        {
            "direction": w(config.width, 1),
            "proposal": w(config.width, 1),
            "diag": w(config.width, 1),
            "lowrank": w(config.width, config.rank),
            "global": w(config.width, 5),
        }
    )
    return p


def init_mlp_model(key: Any, config: ModelConfig = DEFAULT_CONFIG) -> Params:
    """Initialize a shared-coordinate MLP baseline with no attention."""
    k = iter(jax.random.split(key, 9))

    def w(a: int, b: int) -> Any:
        return jax.random.normal(next(k), (a, b)) * a**-0.5

    return {
        "hidden": w(7, config.width),
        "direction": w(config.width, 1),
        "proposal": w(config.width, 1),
        "diag": w(config.width, 1),
        "lowrank": w(config.width, config.rank),
        "global": w(config.width, 5),
    }


def _attention(z: Any, p: Params, i: int, heads: int) -> Any:
    n, width = z.shape
    depth = width // heads

    def split(a: Any) -> Any:
        return a.reshape(n, heads, depth).transpose(1, 0, 2)

    q, k, v = split(z @ p[f"q{i}"]), split(z @ p[f"k{i}"]), split(z @ p[f"v{i}"])
    a = jax.nn.softmax(jnp.einsum("hnd,hmd->hnm", q, k) / jnp.sqrt(depth), axis=-1)
    out = jnp.einsum("hnm,hmd->hnd", a, v).transpose(1, 0, 2).reshape(n, width)
    return z + out @ p[f"o{i}"]


def _layer_norm(z: Any) -> Any:
    mean = jnp.mean(z, axis=-1, keepdims=True)
    variance = jnp.mean((z - mean) ** 2, axis=-1, keepdims=True)
    return (z - mean) * jax.lax.rsqrt(variance + 1e-5)


def apply_model(
    params: Params, features: Any, budget_fraction: Any, config: ModelConfig
) -> dict[str, Any]:
    """Predict direct numerical actions and certification guidance."""
    z = features @ params["embed"]
    task = params["task"].at[0].multiply(budget_fraction)
    z = jnp.concatenate((task, z), axis=0)
    for i in range(config.layers):
        z = jax.nn.gelu(_attention(_layer_norm(z), params, i, config.heads))
    coord, pooled = z[2:], jnp.mean(z[:2], axis=0)
    g = pooled @ params["global"]
    return {
        "direction": jnp.tanh((coord @ params["direction"])[:, 0]),
        "proposal": jnp.tanh((coord @ params["proposal"])[:, 0]),
        "diag": jax.nn.softplus((coord @ params["diag"])[:, 0]) + 1e-4,
        "lowrank": coord @ params["lowrank"] / jnp.sqrt(coord.shape[0]),
        "step": jax.nn.sigmoid(g[0]),
        "exploration": jax.nn.sigmoid(g[1]),
        "region_priority": g[2],
        "split_preference": jax.nn.sigmoid((coord @ params["diag"])[:, 0]),
        "uncertainty": jax.nn.softplus(g[3]),
        "stop": jax.nn.sigmoid(g[4]),
    }


def apply_mlp_model(
    params: Params, features: Any, budget_fraction: Any, config: ModelConfig
) -> dict[str, Any]:
    """Apply the matched-input MLP architectural ablation."""
    budget = jnp.full((features.shape[0], 1), budget_fraction)
    dimension = jnp.full((features.shape[0], 1), jnp.log1p(features.shape[0]) / 5.0)
    coord = jax.nn.gelu(jnp.concatenate((features, budget, dimension), axis=-1) @ params["hidden"])
    pooled = jnp.mean(coord, axis=0)
    g = pooled @ params["global"]
    return {
        "direction": jnp.tanh((coord @ params["direction"])[:, 0]),
        "proposal": jnp.tanh((coord @ params["proposal"])[:, 0]),
        "diag": jax.nn.softplus((coord @ params["diag"])[:, 0]) + 1e-4,
        "lowrank": coord @ params["lowrank"] / jnp.sqrt(coord.shape[0]),
        "step": jax.nn.sigmoid(g[0]),
        "exploration": jax.nn.sigmoid(g[1]),
        "region_priority": g[2],
        "split_preference": jax.nn.sigmoid((coord @ params["diag"])[:, 0]),
        "uncertainty": jax.nn.softplus(g[3]),
        "stop": jax.nn.sigmoid(g[4]),
    }


def precondition(direction: Any, diag: Any, lowrank: Any) -> Any:
    """Compute ``(diag(D) + U U.T) direction`` without a dense matrix."""
    return diag * direction + lowrank @ (lowrank.T @ direction)
