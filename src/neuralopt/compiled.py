"""Compiled engine for the v0.6 ``preconditioned`` loop (Track A: speed, same rule).

The whole loop runs as one ``lax.while_loop`` inside a single ``jit``. The rule is
identical to :func:`neuralopt.optimizer._minimize_preconditioned`:

``u' = clip(u - σ L P ĝ, -1, 1)``; one objective call per candidate; a gradient only
on acceptance (``lax.cond``), so budget accounting stays exact; ``σ`` halves after a
rejection; propose only while ``used + 2 <= budget``.

Errors are returned as status codes and raised by the caller after the loop.
Diagnostics are written to preallocated arrays of length ``budget``. Under ``vmap``
``lax.cond`` evaluates both branches (extra FLOPs), but counters charge a gradient only
when the candidate was accepted.

The Python loop stays the reference and the only path for host/NumPy gradients.

``model_kind="gram"`` (v0.7 Track B, compiled engine only) uses a different candidate
rule, ``u' = clip(u − shrink · σ · P g_u, −1, 1)`` with the un-normalized gradient and the
Gram inverse metric; it has no Python-loop reference and is checked against an
independent harness copy (``scripts/run_gram_gates_v0_7.py``).
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .features import (
    gradient_to_normalized,
    normalize_coordinates,
    parameter_features,
)
from .gram import RejectionInfo, gram_step
from .loop_rules import LOOP_RULES, accept_step
from .model import DEFAULT_CONFIG, ModelConfig, Params, apply_mlp_model, apply_model
from .trajectory import (
    TrajectoryHistory,
    empty_history,
    history_token_features,
    push_history,
    record_rejection,
)

RUNNING, FINISHED, GEOMETRY_ERROR, INITIAL_VALUE_ERROR, INITIAL_GRADIENT_ERROR = range(5)
STATUS_MESSAGES = {
    GEOMETRY_ERROR: "learned geometry is nonfinite or not positive definite",
    INITIAL_VALUE_ERROR: "objective returned a nonfinite initial value",
    INITIAL_GRADIENT_ERROR: "gradient is nonfinite at the initial point",
}


class _State(NamedTuple):
    x: Any
    value: Any
    grad: Any
    history: TrajectoryHistory
    shrink: Any
    used: Any
    objective_evaluations: Any
    gradient_evaluations: Any
    accepted_steps: Any
    rejected_steps: Any
    iteration: Any
    status: Any
    trajectory: Any  # (budget,)
    candidates: Any  # (budget, d)
    candidate_values: Any  # (budget,)
    accepted: Any  # (budget,) bool
    step_lengths: Any  # (budget,)
    diag_min: Any
    diag_max: Any
    lowrank_frobenius: Any
    # Rejection information at the current point (used by the Gram model only).
    rejected_length: Any
    rejected_ratio: Any
    rejected_has: Any
    # Gram model diagnostics per iteration (zero for other models).
    taus: Any  # (budget,)
    sigmas: Any  # (budget,)
    trials: Any  # (budget,) trials per iteration (step-acceptance rule)
    initial_accepted: Any  # (budget,) the model's own step was accepted unchanged
    wolfe_satisfied: Any  # (budget,)


def _solve(
    objective: Any,
    parametric: bool,
    budget: int,
    config: Any,  # ModelConfig, or GramConfig when model_kind == "gram"
    model_kind: str,
    geometry: str,
    structure: str,
    loop_rule: str,
    theta: Any,
    x0: Any,
    lo: Any,
    hi: Any,
    params: Params,
    fixed_diag: Any,
    fixed_step: Any,
) -> tuple[_State, Any]:
    dtype = lo.dtype
    dimension = x0.shape[-1]

    def f(x: Any) -> Any:
        return objective(x, theta) if parametric else objective(x)

    grad_f = jax.grad(f)
    history_on = config.feature_set != "legacy"
    apply = apply_model if model_kind == "transformer" else apply_mlp_model
    leaves = jax.tree_util.tree_leaves(params)
    param_dtype = leaves[0].dtype if leaves else dtype
    if model_kind == "gram" and geometry != "learned":
        raise ValueError("the Gram model is a learned geometry")

    value0 = f(x0).astype(dtype)
    grad0 = grad_f(x0).astype(dtype)
    history = empty_history(config.history_length, dimension, dtype)
    if history_on:
        history = push_history(
            history,
            normalize_coordinates(x0, lo, hi),
            gradient_to_normalized(grad0, lo, hi),
            value0,
        )
    status = jnp.where(
        ~jnp.isfinite(value0),
        INITIAL_VALUE_ERROR,
        jnp.where(jnp.all(jnp.isfinite(grad0)), RUNNING, INITIAL_GRADIENT_ERROR),
    ).astype(jnp.int32)
    zero = jnp.asarray(0, jnp.int32)
    state = _State(
        x=x0,
        value=value0,
        grad=grad0,
        history=history,
        shrink=jnp.asarray(1.0, dtype),
        used=jnp.asarray(2, jnp.int32),
        objective_evaluations=jnp.asarray(1, jnp.int32),
        gradient_evaluations=jnp.asarray(1, jnp.int32),
        accepted_steps=zero,
        rejected_steps=zero,
        iteration=zero,
        status=status,
        trajectory=jnp.full((budget,), jnp.nan, dtype).at[0].set(value0),
        candidates=jnp.zeros((budget, dimension), dtype),
        candidate_values=jnp.full((budget,), jnp.nan, dtype),
        accepted=jnp.zeros((budget,), bool),
        step_lengths=jnp.zeros((budget,), dtype),
        diag_min=jnp.zeros((budget,), dtype),
        diag_max=jnp.zeros((budget,), dtype),
        lowrank_frobenius=jnp.zeros((budget,), dtype),
        rejected_length=jnp.asarray(0.0, dtype),
        rejected_ratio=jnp.asarray(0.0, dtype),
        rejected_has=jnp.asarray(False),
        taus=jnp.zeros((budget,), dtype),
        sigmas=jnp.zeros((budget,), dtype),
        trials=jnp.zeros((budget,), jnp.int32),
        initial_accepted=jnp.zeros((budget,), bool),
        wolfe_satisfied=jnp.zeros((budget,), bool),
    )

    def gradient_norm(grad: Any) -> Any:
        return jnp.linalg.norm(gradient_to_normalized(grad, lo, hi))

    def keep_going(s: _State) -> Any:
        return (s.status == RUNNING) & (s.used + 2 <= budget) & (gradient_norm(s.grad) != 0.0)

    def step(s: _State) -> _State:
        g_u = gradient_to_normalized(s.grad, lo, hi)
        unit = g_u / jnp.linalg.norm(g_u)
        u = normalize_coordinates(s.x, lo, hi)
        if model_kind == "gram":
            rejection = RejectionInfo(s.history.rejections, s.rejected_length,
                                      s.rejected_ratio, s.rejected_has)
            base, sigma, gram_out, tau = gram_step(
                params, config, s.history, g_u, rejection, 1 - s.used.astype(dtype) / budget
            )
            base = base.astype(dtype)
            # Records: step_lengths = α · shrink · σ, taus = τ, sigmas = σ,
            # lowrank_frobenius = ‖A‖_F; diag_min/diag_max are not meaningful here.
            length = sigma.astype(dtype)
            diag = jnp.ones((1,), dtype)
            lowrank = gram_out["A"].astype(dtype)
            gram_tau, gram_sigma = tau.astype(dtype), length
            geometry_ok = (
                jnp.all(jnp.isfinite(base))
                & jnp.isfinite(length)
                & jnp.isfinite(tau)
                & (tau > 0)
                & jnp.all(jnp.isfinite(gram_out["A"]))
                & jnp.all(jnp.isfinite(gram_out["C"]))
            )
        elif geometry == "learned":
            if history_on:
                tokens = history_token_features(s.x, s.grad, lo, hi, s.history)
            else:
                tokens = parameter_features(s.x, s.grad, lo, hi, s.x, config.scale_aware)
            # Match the Python loop, which passes a Python float that adopts the
            # parameters' dtype (float32 checkpoints on float64 problems).
            fraction = (1 - s.used.astype(dtype) / budget).astype(param_dtype)
            out = apply(params, tokens, fraction, config)
            diag, lowrank = out["diag"].astype(dtype), out["lowrank"].astype(dtype)
            if structure == "diagonal":
                lowrank = jnp.zeros_like(lowrank)
            elif structure == "lowrank":
                diag = jnp.ones_like(diag)
            length = (0.5 * out["step"]).astype(dtype)
            geometry_ok = (
                jnp.all(jnp.isfinite(diag))
                & jnp.all(diag > 0)
                & jnp.all(jnp.isfinite(lowrank))
                & jnp.isfinite(length)
            )
        else:
            diag = fixed_diag if geometry == "fixed" else jnp.ones((dimension,), dtype)
            lowrank = jnp.zeros((dimension, 1), dtype)
            length = fixed_step
            geometry_ok = jnp.asarray(True)
        if model_kind != "gram":
            gram_tau = gram_sigma = jnp.asarray(0.0, dtype)
            # Same operations as trajectory.preconditioned_candidate (the v0.6 rule).
            base = diag * unit + lowrank @ (lowrank.T @ unit)

        def evaluate(s: _State) -> _State:
            outcome = accept_step(loop_rule, f, grad_f, s.x, s.value, s.grad, base, length,
                                  s.shrink, budget - s.used, lo, hi)
            accepted = outcome.accepted
            if history_on:
                pushed = push_history(
                    s.history,
                    normalize_coordinates(outcome.x, lo, hi),
                    gradient_to_normalized(outcome.grad, lo, hi),
                    outcome.value,
                )
            else:
                pushed = s.history
            history = jax.tree_util.tree_map(
                lambda a, b: jnp.where(accepted, a, b), pushed, record_rejection(s.history)
            )
            candidate = jnp.where(accepted, outcome.x, outcome.trial_x)
            candidate_value = jnp.where(accepted, outcome.value, outcome.trial_value)
            step_u = normalize_coordinates(outcome.trial_x, lo, hi) - u
            predicted = jnp.sum(g_u * step_u)
            ratio = jnp.where(
                jnp.isfinite(outcome.trial_value) & (predicted != 0),
                (outcome.trial_value - s.value) / jnp.where(predicted != 0, predicted, 1.0),
                10.0,
            )
            units = outcome.objective_evaluations + outcome.gradient_evaluations
            i = s.iteration
            return s._replace(
                x=outcome.x,
                value=outcome.value,
                grad=outcome.grad,
                history=history,
                shrink=outcome.next_shrink.astype(dtype),
                used=s.used + units,
                objective_evaluations=s.objective_evaluations + outcome.objective_evaluations,
                gradient_evaluations=s.gradient_evaluations + outcome.gradient_evaluations,
                accepted_steps=s.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=s.rejected_steps + (~accepted).astype(jnp.int32),
                iteration=i + 1,
                trajectory=s.trajectory.at[i + 1].set(outcome.value),
                candidates=s.candidates.at[i].set(candidate),
                candidate_values=s.candidate_values.at[i].set(candidate_value),
                accepted=s.accepted.at[i].set(accepted),
                step_lengths=s.step_lengths.at[i].set(outcome.applied_scale.astype(dtype)),
                diag_min=s.diag_min.at[i].set(jnp.min(diag)),
                diag_max=s.diag_max.at[i].set(jnp.max(diag)),
                lowrank_frobenius=s.lowrank_frobenius.at[i].set(jnp.linalg.norm(lowrank)),
                rejected_length=jnp.where(accepted, 0.0, jnp.linalg.norm(step_u)).astype(dtype),
                rejected_ratio=jnp.where(accepted, 0.0, jnp.clip(ratio, -10.0, 10.0)).astype(
                    dtype),
                rejected_has=~accepted,
                taus=s.taus.at[i].set(gram_tau),
                sigmas=s.sigmas.at[i].set(gram_sigma),
                trials=s.trials.at[i].set(outcome.trials),
                initial_accepted=s.initial_accepted.at[i].set(outcome.initial_accepted),
                wolfe_satisfied=s.wolfe_satisfied.at[i].set(outcome.wolfe_satisfied),
            )

        def fail(s: _State) -> _State:
            return s._replace(status=jnp.asarray(GEOMETRY_ERROR, jnp.int32))

        return jax.lax.cond(geometry_ok, evaluate, fail, s)

    final = jax.lax.while_loop(keep_going, step, state)
    stationary = (final.status == RUNNING) & (final.used + 2 <= budget)
    final = final._replace(status=jnp.where(final.status == RUNNING, FINISHED, final.status))
    return final, stationary


_STATIC = ("objective", "parametric", "budget", "config", "model_kind", "geometry", "structure",
           "loop_rule")
_solve_jit = jax.jit(_solve, static_argnames=_STATIC)


def _batched(
    objective: Any,
    budget: int,
    config: ModelConfig,
    model_kind: str,
    geometry: str,
    structure: str,
    loop_rule: str,
    thetas: Any,
    x0: Any,
    lo: Any,
    hi: Any,
    params: Params,
    fixed_diag: Any,
    fixed_step: Any,
) -> tuple[_State, Any]:
    solve = partial(_solve, objective, True, budget, config, model_kind, geometry, structure,
                    loop_rule)
    return jax.vmap(solve, in_axes=(0, 0, 0, 0, None, None, None))(
        thetas, x0, lo, hi, params, fixed_diag, fixed_step
    )


_batched_jit = jax.jit(
    _batched, static_argnames=("objective", "budget", "config", "model_kind", "geometry",
                               "structure", "loop_rule")
)


def _geometry_inputs(
    dimension: int, dtype: Any, geometry: str, fixed_metric_diag: Any, preconditioned_step: Any,
    loop_rule: str = "halving",
) -> tuple[Any, Any]:
    if loop_rule not in LOOP_RULES:
        raise ValueError(f"loop_rule must be one of {LOOP_RULES}")
    if geometry not in {"learned", "fixed", "identity"}:
        raise ValueError("geometry must be 'learned', 'fixed', or 'identity'")
    if geometry != "learned" and (preconditioned_step is None or not preconditioned_step > 0):
        raise ValueError("non-learned preconditioned geometry requires preconditioned_step > 0")
    diag = (
        jnp.asarray(fixed_metric_diag, dtype)
        if geometry == "fixed" and fixed_metric_diag is not None
        else jnp.ones((dimension,), dtype)
    )
    step = jnp.asarray(preconditioned_step if preconditioned_step is not None else 0.0, dtype)
    return diag, step


def preconditioned_solve(
    objective: Any,
    x0: Any,
    bounds: tuple[Any, Any],
    budget: int,
    *,
    theta: Any = None,
    params: Params | None = None,
    config: ModelConfig = DEFAULT_CONFIG,
    model_kind: str = "transformer",
    geometry: str = "learned",
    structure: str = "full",
    fixed_metric_diag: Any = None,
    preconditioned_step: float | None = None,
    loop_rule: str = "halving",
) -> dict[str, Any]:
    """Run one compiled solve; returns host arrays (see :class:`_State`) plus status.

    ``objective`` must be JAX-traceable: ``objective(x)``, or ``objective(x, theta)``
    when ``theta`` is given. It is a static argument: reuse the same function object
    to reuse the compilation.
    """
    if budget < 2:
        raise ValueError("preconditioned loop needs a budget of at least 2 (objective + gradient)")
    lo, hi = (jnp.asarray(b) for b in bounds)
    x0 = jnp.asarray(x0, lo.dtype)
    diag, step = _geometry_inputs(x0.shape[-1], lo.dtype, geometry, fixed_metric_diag,
                                  preconditioned_step, loop_rule)
    state, stationary = _solve_jit(
        objective=objective,
        parametric=theta is not None,
        budget=budget,
        config=config,
        model_kind=model_kind,
        geometry=geometry,
        structure=structure,
        loop_rule=loop_rule,
        theta=theta,
        x0=x0,
        lo=lo,
        hi=hi,
        params=params if params is not None else {},
        fixed_diag=diag,
        fixed_step=step,
    )
    return _to_host(state, stationary)


def preconditioned_solve_batch(
    objective: Any,
    thetas: Any,
    x0: Any,
    bounds: tuple[Any, Any],
    budget: int,
    *,
    params: Params | None = None,
    config: ModelConfig = DEFAULT_CONFIG,
    model_kind: str = "transformer",
    geometry: str = "learned",
    structure: str = "full",
    fixed_metric_diag: Any = None,
    preconditioned_step: float | None = None,
    loop_rule: str = "halving",
) -> dict[str, Any]:
    """``vmap`` the compiled solve over tasks ``objective(x, theta_i)`` of one dimension.

    ``thetas``, ``x0`` and both bounds carry a leading task axis. Errors are *not*
    raised: inspect ``status`` per task (see :data:`STATUS_MESSAGES`).
    """
    lo, hi = (jnp.asarray(b) for b in bounds)
    x0 = jnp.asarray(x0, lo.dtype)
    diag, step = _geometry_inputs(x0.shape[-1], lo.dtype, geometry, fixed_metric_diag,
                                  preconditioned_step, loop_rule)
    state, stationary = _batched_jit(
        objective=objective,
        budget=budget,
        config=config,
        model_kind=model_kind,
        geometry=geometry,
        structure=structure,
        loop_rule=loop_rule,
        thetas=thetas,
        x0=x0,
        lo=lo,
        hi=hi,
        params=params if params is not None else {},
        fixed_diag=diag,
        fixed_step=step,
    )
    return _to_host(state, stationary)


def _to_host(state: _State, stationary: Any) -> dict[str, Any]:
    host = jax.device_get(state._asdict())
    host["history"] = TrajectoryHistory(*jax.device_get(tuple(state.history)))
    host["stationary"] = jax.device_get(stationary)
    host["fun"] = host["value"]
    return host
