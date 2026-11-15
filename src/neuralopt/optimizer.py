"""Learned inference loop and public minimize function."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .certification import Certificate, Expr, certify
from .features import gradient_to_normalized, parameter_features
from .model import (
    DEFAULT_CONFIG,
    ModelConfig,
    Params,
    apply_mlp_model_jit,
    apply_model_jit,
    precondition,
)
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


def _cosine_alignment(direction: np.ndarray, gradient: np.ndarray) -> float:
    """Cosine with negative gradient for vectors in the same coordinates."""
    denominator = float(np.linalg.norm(direction) * np.linalg.norm(gradient))
    return 0.0 if denominator <= 1e-12 else float(direction @ (-gradient) / denominator)


def minimize(
    problem: Problem,
    mode: Literal["learned", "certified"] = "learned",
    max_evaluations: int | None = None,
    seed: int = 0,
    params: Params | None = None,
    config: ModelConfig = DEFAULT_CONFIG,
    absolute_gap: float = 1e-6,
    record_diagnostics: bool = False,
    model_kind: Literal["transformer", "mlp"] = "transformer",
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
            np.asarray(cert.incumbent_x, dtype=np.float64),
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
    if problem.gradient is None:
        objective_fn = jax.jit(problem.objective)
        grad_fn = jax.jit(jax.grad(problem.objective))
    else:
        # Explicit gradients support host/NumPy objectives that cannot be traced.
        objective_fn = problem.objective
        grad_fn = problem.gradient
    x, best_x = problem.x0, problem.x0
    assert x is not None and best_x is not None  # Guaranteed by Problem validation.
    value = float(objective_fn(x))
    evaluations = 1
    if not np.isfinite(value):
        raise FloatingPointError("objective returned a nonfinite initial value")
    best, history = value, [value]
    proposals = 0
    records: list[dict[str, Any]] = []
    while evaluations < budget:
        x_before, value_before = x, value
        try:
            grad = grad_fn(x)
        except Exception as exc:
            raise ValueError("objective is not differentiable; provide a gradient") from exc
        apply = apply_model_jit if model_kind == "transformer" else apply_mlp_model_jit
        out = apply(
            params, parameter_features(x, grad, lo, hi, best_x), 1 - evaluations / budget, config
        )
        raw_direction = out["direction"]
        direction = precondition(raw_direction, out["diag"], out["lowrank"])
        scale = 0.25 * out["step"] * width
        unprojected = x + scale * direction / (jnp.linalg.norm(direction) + 1e-8)
        candidate = jnp.clip(unprojected, lo, hi)
        candidate_kind = "update"
        # The proposal head is independently evaluated on alternating iterations.
        if evaluations % 2 == 0:
            unprojected = (lo + hi) / 2 + 0.5 * width * out["proposal"]
            candidate = jnp.clip(unprojected, lo, hi)
            proposals += 1
            candidate_kind = "proposal"
        candidate_value = float(objective_fn(candidate))
        evaluations += 1
        finite = bool(np.isfinite(candidate_value))
        accepted = finite and candidate_value < value
        improved_best = finite and candidate_value < best
        if improved_best:
            x, best_x, best = candidate, candidate, candidate_value
            value = candidate_value
        elif accepted:
            x, value = candidate, candidate_value
        # Rejected candidates leave x and its corresponding value unchanged.
        if record_diagnostics:
            g = np.asarray(grad, dtype=float)
            normalized_g = np.asarray(gradient_to_normalized(grad, lo, hi), dtype=float)
            raw = np.asarray(raw_direction, dtype=float)
            pre = np.asarray(direction, dtype=float)
            applied = np.asarray(candidate - x_before, dtype=float)
            gnorm = float(np.linalg.norm(g))

            diag = np.asarray(out["diag"], dtype=float)
            lowrank = np.asarray(out["lowrank"], dtype=float)
            dense = np.diag(diag) + lowrank @ lowrank.T
            eig = np.linalg.eigvalsh(dense)
            displacement = np.asarray(candidate - np.asarray(x_before), dtype=float)
            records.append(
                {
                    "evaluation": evaluations,
                    "kind": candidate_kind,
                    "objective_before": float(value_before),
                    "candidate_objective": candidate_value,
                    "best_objective": float(best),
                    "accepted": bool(accepted),
                    "improved_best": bool(improved_best),
                    "reason": "nonfinite"
                    if not finite
                    else "improved"
                    if accepted
                    else "no_improvement",
                    "gradient_norm": gnorm,
                    "raw_alignment": _cosine_alignment(raw, normalized_g),
                    "preconditioned_alignment": _cosine_alignment(pre, normalized_g),
                    "applied_alignment": _cosine_alignment(applied, g),
                    "directional_derivative": float(normalized_g @ pre),
                    "step_length": float(np.linalg.norm(np.asarray(unprojected - x_before))),
                    "effective_displacement": float(np.linalg.norm(displacement)),
                    "projection_norm": float(np.linalg.norm(np.asarray(unprojected - candidate))),
                    "projected_fraction": float(np.mean(np.asarray(unprojected != candidate))),
                    "boundary_fraction": float(
                        np.mean(
                            np.isclose(np.asarray(candidate), np.asarray(lo))
                            | np.isclose(np.asarray(candidate), np.asarray(hi))
                        )
                    ),
                    "relative_improvement": float(
                        (history[-1] - best) / max(1.0, abs(history[-1]))
                    ),
                    "preconditioner_min_eigenvalue": float(eig[0]),
                    "preconditioner_max_eigenvalue": float(eig[-1]),
                    "preconditioner_condition": float(eig[-1] / eig[0]),
                    "diag_min": float(diag.min()),
                    "diag_max": float(diag.max()),
                    "lowrank_frobenius": float(np.linalg.norm(lowrank)),
                    "candidate": np.asarray(candidate).tolist(),
                }
            )
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
            "records": records if record_diagnostics else None,
            "acceptance_rate": float(np.mean([r["accepted"] for r in records]))
            if records
            else None,
        },
    )
