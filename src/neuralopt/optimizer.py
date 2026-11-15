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
from .hamiltonian import DEFAULT_HAMILTONIAN_CONFIG, HamiltonianConfig, hamiltonian_proposal
from .langevin import (
    DEFAULT_OVERDAMPED_CONFIG,
    DEFAULT_UNDERDAMPED_CONFIG,
    OverdampedConfig,
    UnderdampedConfig,
    overdamped_transition,
    underdamped_transition,
)
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


def _validate_policy_params(params: Params, model_kind: str, config: ModelConfig) -> None:
    common = {"direction", "proposal", "diag", "lowrank", "global"}
    if model_kind == "mlp":
        required = common | {"hidden"}
    else:
        attention = {
            f"{name}{layer}" for layer in range(config.layers) for name in ("q", "k", "v", "o")
        }
        required = common | {"embed", "task"} | attention
    missing = sorted(required - params.keys())
    if missing:
        raise ValueError(f"{model_kind} parameters are missing keys: {', '.join(missing)}")


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
    candidate_method: Literal["direct", "hamiltonian", "langevin", "underdamped"] = "direct",
    hamiltonian_config: HamiltonianConfig = DEFAULT_HAMILTONIAN_CONFIG,
    hamiltonian_metric: Literal["identity", "fixed", "learned"] = "learned",
    fixed_metric_diag: Any | None = None,
    overdamped_config: OverdampedConfig = DEFAULT_OVERDAMPED_CONFIG,
    underdamped_config: UnderdampedConfig = DEFAULT_UNDERDAMPED_CONFIG,
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
    if model_kind not in {"transformer", "mlp"}:
        raise ValueError("model_kind must be 'transformer' or 'mlp'")
    if candidate_method not in {"direct", "hamiltonian", "langevin", "underdamped"}:
        raise ValueError(
            "candidate_method must be 'direct', 'hamiltonian', 'langevin', or 'underdamped'"
        )
    if hamiltonian_metric not in {"identity", "fixed", "learned"}:
        raise ValueError("hamiltonian_metric must be 'identity', 'fixed', or 'learned'")
    if candidate_method != "direct" and problem.constraints:
        raise ValueError("stochastic proposals currently support box constraints only")
    checkpoint_name = "caller-provided"
    if params is None:
        if model_kind != "transformer":
            raise ValueError("model_kind='mlp' requires caller-provided MLP parameters")
        from .training import load_checkpoint

        resource = files("neuralopt").joinpath("checkpoints/demo-v0.1.0.npz")
        with as_file(resource) as checkpoint_path:
            params, config = load_checkpoint(checkpoint_path)
        checkpoint_name = "bundled-demo-v0.1.0"
    _validate_policy_params(params, model_kind, config)
    lo, hi = problem.bounds
    if hamiltonian_metric == "fixed":
        if fixed_metric_diag is None:
            raise ValueError("fixed_metric_diag is required when hamiltonian_metric='fixed'")
        fixed_metric_diag = jnp.asarray(fixed_metric_diag, dtype=lo.dtype)
        if fixed_metric_diag.shape != (problem.dimension,) or not bool(
            jnp.all(jnp.isfinite(fixed_metric_diag) & (fixed_metric_diag > 0))
        ):
            raise ValueError(
                "fixed_metric_diag must be a finite positive vector of problem dimension"
            )
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
    gradient_evaluations = 0
    hmc_transitions = 0
    hmc_accepted = 0
    leapfrog_steps = 0
    stochastic_integration_steps = 0
    hmc_unconstrained_state: Any | None = None
    stochastic_momentum: Any | None = None
    records: list[dict[str, Any]] = []
    key = jax.random.PRNGKey(seed)
    while evaluations < budget:
        if candidate_method == "hamiltonian":
            proposal_cost = hamiltonian_config.leapfrog_steps + 4
        elif candidate_method == "langevin":
            proposal_cost = overdamped_config.steps + 1
        elif candidate_method == "underdamped":
            proposal_cost = 2 * underdamped_config.steps + 1
        else:
            proposal_cost = 1
        if evaluations + proposal_cost > budget:
            break
        x_before, value_before = x, value
        try:
            grad = grad_fn(x)
            gradient_evaluations += 1
        except Exception as exc:
            raise ValueError("objective is not differentiable; provide a gradient") from exc
        apply = apply_model_jit if model_kind == "transformer" else apply_mlp_model_jit
        out = apply(
            params, parameter_features(x, grad, lo, hi, best_x), 1 - evaluations / budget, config
        )
        raw_direction = out["direction"]
        direction = precondition(raw_direction, out["diag"], out["lowrank"])
        energy_error = None
        divergent = False
        if candidate_method in {"hamiltonian", "langevin", "underdamped"}:
            key, proposal_key = jax.random.split(key)
            if hamiltonian_metric == "identity":
                mass_diag = jnp.ones_like(out["diag"])
                mass_lowrank = jnp.zeros_like(out["lowrank"])
            elif hamiltonian_metric == "fixed":
                mass_diag = fixed_metric_diag
                mass_lowrank = jnp.zeros_like(out["lowrank"])
            else:
                mass_diag, mass_lowrank = out["diag"], out["lowrank"]
            proposal: Any
            try:
                if candidate_method == "hamiltonian":
                    proposal = hamiltonian_proposal(
                        proposal_key,
                        x,
                        problem.objective,
                        lo,
                        hi,
                        mass_diag,
                        mass_lowrank,
                        hamiltonian_config,
                        unconstrained_state=hmc_unconstrained_state,
                    )
                elif candidate_method == "langevin":
                    proposal = overdamped_transition(
                        proposal_key,
                        x,
                        problem.objective,
                        lo,
                        hi,
                        mass_diag,
                        mass_lowrank,
                        overdamped_config,
                        unconstrained_state=hmc_unconstrained_state,
                    )
                else:
                    proposal = underdamped_transition(
                        proposal_key,
                        x,
                        problem.objective,
                        lo,
                        hi,
                        mass_diag,
                        mass_lowrank,
                        underdamped_config,
                        unconstrained_state=hmc_unconstrained_state,
                        momentum=stochastic_momentum,
                    )
            except Exception as exc:
                raise ValueError(
                    "stochastic proposals require a JAX-differentiable objective"
                ) from exc
            candidate = proposal.candidate if candidate_method == "hamiltonian" else proposal.state
            unprojected = candidate
            candidate_value = float(proposal.candidate_value)
            accepted = (
                bool(proposal.accepted)
                if candidate_method == "hamiltonian"
                else bool(proposal.finite)
            )
            energy_error = (
                float(proposal.energy_error) if candidate_method == "hamiltonian" else None
            )
            divergent = (
                bool(proposal.divergent)
                if candidate_method == "hamiltonian"
                else not bool(proposal.finite)
            )
            # Never poison a stochastic trajectory with a nonfinite internal
            # position or momentum. HMC already returns the prior chain state
            # after rejection, so its internal state is always safe to retain.
            if candidate_method == "hamiltonian" or accepted:
                hmc_unconstrained_state = proposal.unconstrained_state
                if candidate_method == "underdamped":
                    stochastic_momentum = proposal.momentum
            candidate_kind = candidate_method
            evaluations += proposal.objective_evaluations
            gradient_evaluations += proposal.gradient_evaluations
            if candidate_method == "hamiltonian":
                leapfrog_steps += proposal.leapfrog_steps
                hmc_transitions += 1
                hmc_accepted += int(accepted)
            else:
                stochastic_integration_steps += proposal.integration_steps
        else:
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
            accepted = bool(np.isfinite(candidate_value) and candidate_value < value)
        finite = bool(np.isfinite(candidate_value))
        feasible = bool(
            np.all(np.asarray(candidate) >= np.asarray(lo))
            and np.all(np.asarray(candidate) <= np.asarray(hi))
        )
        normalized_distance = float(
            np.linalg.norm(np.asarray((candidate - x_before) / width, dtype=float))
        )
        duplicate = normalized_distance <= 1e-8
        objective_improved = finite and candidate_value < value_before
        improved_best = finite and candidate_value < best
        if improved_best:
            best_x, best = candidate, candidate_value
        if accepted:
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
                    "feasible": feasible,
                    "duplicate": duplicate,
                    "objective_improved": bool(objective_improved),
                    "improved_best": bool(improved_best),
                    "reason": (
                        "nonfinite"
                        if not finite
                        else "infeasible"
                        if not feasible
                        else "divergent"
                        if divergent
                        else "accepted"
                        if accepted
                        else "acceptance_threshold"
                        if candidate_method == "hamiltonian"
                        else "objective_worsening"
                    ),
                    "gradient_norm": gnorm,
                    "raw_alignment": _cosine_alignment(raw, normalized_g),
                    "preconditioned_alignment": _cosine_alignment(pre, normalized_g),
                    "applied_alignment": _cosine_alignment(applied, g),
                    "directional_derivative": float(normalized_g @ pre),
                    "step_length": float(np.linalg.norm(np.asarray(unprojected - x_before))),
                    "effective_displacement": float(np.linalg.norm(displacement)),
                    "normalized_candidate_distance": normalized_distance,
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
                    "energy_error": energy_error,
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
            "candidate_method": candidate_method,
            "hamiltonian_metric": hamiltonian_metric if hmc_transitions else None,
            "stochastic_geometry": hamiltonian_metric if candidate_method != "direct" else None,
            "candidate_proposals": len(history) - 1,
            "objective_evaluations": evaluations,
            "gradient_evaluations": gradient_evaluations,
            "leapfrog_steps": leapfrog_steps,
            "stochastic_integration_steps": stochastic_integration_steps,
            "hmc_transitions": hmc_transitions,
            "hmc_transition_acceptance": hmc_accepted / hmc_transitions
            if hmc_transitions
            else None,
            "records": records if record_diagnostics else None,
            "acceptance_rate": float(np.mean([r["accepted"] for r in records]))
            if records
            else None,
            "objective_improvement_rate": float(np.mean([r["objective_improved"] for r in records]))
            if records
            else None,
            "incumbent_update_rate": float(np.mean([r["improved_best"] for r in records]))
            if records
            else None,
            "feasible_candidate_rate": float(np.mean([r["feasible"] for r in records]))
            if records
            else None,
            "duplicate_candidate_rate": float(np.mean([r["duplicate"] for r in records]))
            if records
            else None,
        },
    )
