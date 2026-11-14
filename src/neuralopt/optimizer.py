"""Learned inference loop and public minimize function."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .certification import Certificate, Expr, certify
from .features import parameter_features
from .model import DEFAULT_CONFIG, ModelConfig, Params, apply_model, precondition
from .problem import Problem


@dataclass(frozen=True)
class OptimizationResult:
    x: Any
    fun: float
    status: str
    evaluations: int
    trajectory: tuple[float, ...]
    lower_bound: float | None = None
    upper_bound: float | None = None
    certificate: Certificate | None = None
    diagnostics: dict[str, Any] | None = None


def minimize(
    problem: Problem,
    mode: Literal["learned", "certified"] = "learned",
    max_evaluations: int | None = None,
    seed: int = 0,
    params: Params | None = None,
    config: ModelConfig = DEFAULT_CONFIG,
    absolute_gap: float = 1e-6,
) -> OptimizationResult:
    """Minimize a bounded objective with learned actions or rigorous certification."""
    budget = max_evaluations or problem.max_evaluations
    if budget < 1:
        raise ValueError("max_evaluations must be positive")
    if mode == "certified":
        if not isinstance(problem.certified_expression, Expr):
            raise ValueError(
                "certified mode requires a neuralopt Expr; arbitrary JAX objectives are unsupported"
            )
        cert = certify(
            problem.certified_expression,
            *problem.bounds,
            absolute_gap,
            budget,
            problem.identifier or "anonymous",
        )
        return OptimizationResult(
            jnp.asarray(cert.incumbent_x),
            cert.upper_bound,
            cert.status,
            cert.nodes_processed,
            (),
            cert.lower_bound,
            cert.upper_bound,
            cert,
        )
    if mode != "learned":
        raise ValueError("mode must be 'learned' or 'certified'")
    checkpoint_name = "caller-provided"
    if params is None:
        from .training import load_checkpoint

        resource = files("neuralopt").joinpath("checkpoints/demo-v0.1.0.npz")
        with as_file(resource) as checkpoint_path:
            params, config = load_checkpoint(checkpoint_path)
        checkpoint_name = "bundled-demo-v0.1.0"
    lo, hi = problem.bounds
    width = hi - lo
    x, best_x = problem.x0, problem.x0
    assert x is not None and best_x is not None  # Guaranteed by Problem validation.
    value = float(problem.objective(x))
    evaluations = 1
    if not np.isfinite(value):
        raise FloatingPointError("objective returned a nonfinite initial value")
    best, history = value, [value]
    grad_fn = problem.gradient or jax.grad(problem.objective)
    proposals = 0
    while evaluations < budget:
        try:
            grad = grad_fn(x)
        except Exception as exc:
            raise ValueError("objective is not differentiable; provide a gradient") from exc
        out = apply_model(
            params, parameter_features(x, grad, lo, hi, best_x), 1 - evaluations / budget, config
        )
        direction = precondition(out["direction"], out["diag"], out["lowrank"])
        scale = 0.25 * out["step"] * width
        candidate = jnp.clip(x + scale * direction / (jnp.linalg.norm(direction) + 1e-8), lo, hi)
        # The proposal head is independently evaluated on alternating iterations.
        if evaluations % 2 == 0:
            candidate = jnp.clip((lo + hi) / 2 + 0.5 * width * out["proposal"], lo, hi)
            proposals += 1
        candidate_value = float(problem.objective(candidate))
        evaluations += 1
        if np.isfinite(candidate_value) and candidate_value < best:
            x, best_x, best = candidate, candidate, candidate_value
        elif np.isfinite(candidate_value):
            # Fixed rejection is numerical protection, not a classical fallback.
            x = jnp.clip((x + best_x) / 2, lo, hi)
        history.append(best)
    return OptimizationResult(
        best_x,
        best,
        "budget_exhausted",
        evaluations,
        tuple(history),
        diagnostics={
            "independent_proposals": proposals,
            "parameter_count": sum(v.size for v in jax.tree_util.tree_leaves(params)),
            "checkpoint": checkpoint_name,
            "seed": seed,
        },
    )
