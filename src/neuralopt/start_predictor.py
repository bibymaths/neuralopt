"""v0.8 start predictor ``g_φ``: a masked set encoder over observation tokens that proposes
K starting points (inside the box) and a confidence for each.

Frozen design: ``docs/research/v0.8-preregistration.md`` §5. Refinement is NeuralOpt's own
L-BFGS (:mod:`neuralopt.lbfgs`); the predictor never evaluates the objective at inference.

Tokens, one per observation: ``(t / 10, standardized y, species one-hot (2), log σ, dose)``;
padding is masked. Encoder: linear embedding to ``width``, ``layers`` pre-layer-norm
transformer blocks (masked self-attention, GELU MLP), then masked mean pooling concatenated
with attention pooling (one learned query). Heads: K candidates mapped into the box with a
scaled ``tanh``; a confidence MLP per candidate (parameters prefixed ``conf_`` so that stage
2 can train them alone).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

START_SCHEMA = "neuralopt-start/v0.8-v1"
TOKEN_FEATURES = 6
SPECIES = 2


@dataclass(frozen=True)
class StartConfig:
    dimension: int = 4
    candidates: int = 8
    width: int = 64
    heads: int = 4
    layers: int = 2
    mlp_width: int = 128
    head_width: int = 128
    confidence_width: int = 64

    def __post_init__(self) -> None:
        if self.width % self.heads or self.layers < 1 or self.candidates < 1:
            raise ValueError("width must be divisible by heads; layers and candidates >= 1")


# ---- tokens ---------------------------------------------------------------------------


def context_stats(contexts: dict[str, Any]) -> dict[str, Any]:
    """Per-species mean and standard deviation of the observed values (training split)."""
    y, mask, species = (np.asarray(contexts[k]) for k in ("y", "mask", "species"))
    mean, std = np.zeros(SPECIES), np.ones(SPECIES)
    for s in range(SPECIES):
        values = y[mask & (species == s)]
        if values.size:
            mean[s] = values.mean()
            std[s] = max(float(values.std()), 1e-12)
    return {"y_mean": jnp.asarray(mean), "y_std": jnp.asarray(std)}


def observation_tokens(context: dict[str, Any], stats: dict[str, Any]) -> Any:
    m = context["mask"].astype(context["y"].dtype)
    s = context["species"]
    y = (context["y"] - stats["y_mean"][s]) / stats["y_std"][s]
    dose = jnp.broadcast_to(context["dose"], context["t"].shape)
    tokens = jnp.stack([context["t"] / 10.0, y, (s == 0).astype(y.dtype),
                        (s == 1).astype(y.dtype), jnp.log(context["sigma"]), dose], axis=-1)
    return tokens * m[:, None]


# ---- model ----------------------------------------------------------------------------


def init_start_model(key: Any, config: StartConfig) -> dict[str, Any]:
    keys = iter(jax.random.split(key, 16 + 8 * config.layers))
    w, d, k = config.width, config.dimension, config.candidates

    def weight(a: int, b: int, scale: float = 1.0) -> Any:
        return jax.random.normal(next(keys), (a, b)) * scale * a**-0.5

    params: dict[str, Any] = {"embed": weight(TOKEN_FEATURES, w), "embed_b": jnp.zeros(w)}
    for i in range(config.layers):
        for name in ("q", "k", "v", "o"):
            params[f"{name}{i}"] = weight(w, w)
        params[f"mlp_in{i}"] = weight(w, config.mlp_width)
        params[f"mlp_in_b{i}"] = jnp.zeros(config.mlp_width)
        params[f"mlp_out{i}"] = weight(config.mlp_width, w)
        params[f"mlp_out_b{i}"] = jnp.zeros(w)
    params["pool_query"] = jax.random.normal(next(keys), (w,)) * w**-0.5
    params["pool_key"] = weight(w, w)
    params["cand_in"] = weight(2 * w, config.head_width)
    params["cand_in_b"] = jnp.zeros(config.head_width)
    params["cand_out"] = weight(config.head_width, k * d, 0.1)
    # distinct initial candidates (spread over the box) so best-of-K starts diverse
    params["cand_out_b"] = jax.random.normal(next(keys), (k * d,)) * 0.5
    params["conf_in"] = weight(2 * w + d, config.confidence_width)
    params["conf_in_b"] = jnp.zeros(config.confidence_width)
    params["conf_out"] = weight(config.confidence_width, 1, 0.1)
    params["conf_out_b"] = jnp.zeros(1)
    return params


def _layer_norm(z: Any) -> Any:
    mean = jnp.mean(z, axis=-1, keepdims=True)
    variance = jnp.mean((z - mean) ** 2, axis=-1, keepdims=True)
    return (z - mean) * jax.lax.rsqrt(variance + 1e-5)


def _attention(z: Any, params: dict[str, Any], i: int, heads: int, valid: Any) -> Any:
    n, width = z.shape
    depth = width // heads

    def split(a: Any) -> Any:
        return a.reshape(n, heads, depth).transpose(1, 0, 2)

    q, k, v = (split(z @ params[f"{name}{i}"]) for name in ("q", "k", "v"))
    logits = jnp.einsum("hnd,hmd->hnm", q, k) / jnp.sqrt(depth)
    logits = jnp.where(valid[None, None, :], logits, -1e30)
    out = jnp.einsum("hnm,hmd->hnd", jax.nn.softmax(logits, axis=-1), v)
    return out.transpose(1, 0, 2).reshape(n, width) @ params[f"o{i}"]


def encode(params: dict[str, Any], tokens: Any, mask: Any, config: StartConfig) -> Any:
    """Pooled representation (2 · width) of one instance's observation set."""
    z = tokens @ params["embed"] + params["embed_b"]
    for i in range(config.layers):
        z = z + _attention(_layer_norm(z), params, i, config.heads, mask)
        h = jax.nn.gelu(_layer_norm(z) @ params[f"mlp_in{i}"] + params[f"mlp_in_b{i}"])
        z = z + h @ params[f"mlp_out{i}"] + params[f"mlp_out_b{i}"]
    z = _layer_norm(z)
    m = mask.astype(z.dtype)
    mean = jnp.sum(z * m[:, None], axis=0) / jnp.maximum(jnp.sum(m), 1.0)
    scores = (z @ params["pool_key"]) @ params["pool_query"] / jnp.sqrt(config.width)
    weights = jax.nn.softmax(jnp.where(mask, scores, -1e30))
    return jnp.concatenate([mean, weights @ z])


def _to_u(x: Any, lo: Any, hi: Any) -> Any:
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def start_candidates(params: dict[str, Any], pooled: Any, lo: Any, hi: Any,
                     config: StartConfig) -> Any:
    h = jax.nn.gelu(pooled @ params["cand_in"] + params["cand_in_b"])
    raw = (h @ params["cand_out"] + params["cand_out_b"]).reshape(config.candidates,
                                                                  config.dimension)
    u = jnp.clip(jnp.tanh(raw), -1.0 + 1e-12, 1.0 - 1e-12)
    return lo + (hi - lo) * (u + 1.0) / 2.0


def start_confidence(params: dict[str, Any], pooled: Any, candidates: Any, lo: Any,
                     hi: Any) -> Any:
    inputs = jnp.concatenate([jnp.broadcast_to(pooled, (candidates.shape[0], pooled.shape[0])),
                              _to_u(candidates, lo, hi)], axis=1)
    h = jax.nn.gelu(inputs @ params["conf_in"] + params["conf_in_b"])
    return jax.nn.sigmoid((h @ params["conf_out"] + params["conf_out_b"])[:, 0])


def predict_starts(params: dict[str, Any], context: dict[str, Any], stats: dict[str, Any],
                   lo: Any, hi: Any, config: StartConfig) -> dict[str, Any]:
    """K candidate starts (inside the box), their confidences, and the order by confidence."""
    lo, hi = jnp.asarray(lo), jnp.asarray(hi)
    pooled = encode(params, observation_tokens(context, stats), context["mask"], config)
    candidates = start_candidates(params, pooled, lo, hi, config)
    confidence = start_confidence(params, jax.lax.stop_gradient(pooled),
                                  jax.lax.stop_gradient(candidates), lo, hi)
    return {"candidates": candidates, "confidence": confidence,
            "order": jnp.argsort(-confidence)}


# ---- losses and diagnostics ------------------------------------------------------------


def regression_loss(candidates: Any, target: Any, lo: Any, hi: Any) -> Any:
    """Best-of-K squared distance to the target optimum, in box-normalized coordinates."""
    diff = _to_u(candidates, lo, hi) - _to_u(target, lo, hi)
    return jnp.min(jnp.sum(diff * diff, axis=-1))


def objective_loss(candidates: Any, objective: Any, context: Any, f_scale: Any) -> Any:
    """Best-of-K objective value, scaled by ``f_scale`` (no unrolled steps)."""
    values = jax.vmap(lambda x: objective(x, context))(candidates)
    return jnp.min(values) / f_scale


def candidate_diversity(candidates: Any, lo: Any, hi: Any) -> Any:
    """Median pairwise distance between candidates in box-normalized coordinates."""
    u = _to_u(jnp.asarray(candidates), jnp.asarray(lo), jnp.asarray(hi))
    k = u.shape[0]
    i, j = np.triu_indices(k, 1)
    return jnp.median(jnp.linalg.norm(u[i] - u[j], axis=-1))


# ---- checkpoints (fail closed) ---------------------------------------------------------


def save_start_checkpoint(path: str | Path, params: dict[str, Any], config: StartConfig,
                          stats: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
    arrays = {f"param__{k}": np.asarray(v) for k, v in params.items()}
    arrays.update({f"stat__{k}": np.asarray(v) for k, v in stats.items()})
    arrays["schema"] = np.array([START_SCHEMA])
    arrays["start_config"] = np.array([json.dumps(asdict(config), sort_keys=True)])
    arrays["metadata"] = np.array([json.dumps(metadata or {}, sort_keys=True, default=str)])
    saver: Any = np.savez_compressed
    saver(path, **arrays)


def load_start_checkpoint(path: str | Path) -> tuple[dict[str, Any], StartConfig,
                                                     dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"][0]) if "schema" in data.files else None
        if schema != START_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema for the start predictor: {schema}")
        config = StartConfig(**json.loads(str(data["start_config"][0])))
        params = {k.removeprefix("param__"): jnp.asarray(data[k]) for k in data.files
                  if k.startswith("param__")}
        stats = {k.removeprefix("stat__"): jnp.asarray(data[k]) for k in data.files
                 if k.startswith("stat__")}
    expected = init_start_model(jax.random.PRNGKey(0), config)
    if {k: v.shape for k, v in params.items()} != {k: v.shape for k, v in expected.items()}:
        raise ValueError("start checkpoint parameters do not match its schema configuration")
    return params, config, stats
