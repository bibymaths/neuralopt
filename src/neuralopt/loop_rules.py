"""Step-acceptance rules for the preconditioned loop (v0.7, opt-in ``loop_rule``).

A step-acceptance rule decides whether, and how far, to move along the displacement the
model proposes. It never builds curvature: there is no BFGS update and no classical
solver here. The model's step is always the first trial (``α = 1``, times the fallback
``shrink``); a rule may only shorten or lengthen it along the same direction.

The trial at ``α`` is ``u(α) = u − (α · shrink · step) · base`` in normalized
coordinates, where ``base`` is the model's direction (``P ĝ`` or ``P g_u``) and ``step``
its scalar length (``0.5 · step_head`` or ``σ``).

* ``"halving"`` (v0.6 rule, default): one trial, clipped to the box; accept any finite
  decrease. A rejection halves ``shrink`` for the next iteration. Costs 1 objective,
  plus 1 gradient if the trial improved.
* ``"armijo"``: sufficient decrease ``f(u(α)) ≤ f + c1 g_uᵀ Δu`` (with strict decrease)
  on the clipped path, with safeguarded quadratic-interpolation backtracking
  (``α ← clip(α_q, 0.1α, 0.5α)``) and no curvature condition. Each trial costs 1
  objective; the accepted point costs 1 gradient.
* ``"wolfe"``: strong Wolfe conditions with bracketing and zoom (Nocedal & Wright,
  Algorithms 3.5/3.6), allowing lengthening up to ``α = 10`` and never leaving the box
  (``α`` is capped at the box boundary). Each trial costs 1 objective + 1 gradient. If
  the budget or trial limit ends the search, the best decreasing trial is accepted (its
  gradient is known); ``wolfe_satisfied`` records whether the conditions held. A
  non-descent or zero-length direction falls back to one ``"halving"`` trial.

For ``"armijo"`` and ``"wolfe"``, a failed search (no acceptable trial) halves
``shrink`` for the next iteration, so halving remains only as the fallback after
rejection. Rejected trials never form secant pairs.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .features import denormalize_coordinates, gradient_to_normalized, normalize_coordinates

LOOP_RULES = ("halving", "armijo", "wolfe")
C1, C2 = 1e-4, 0.9
MAX_TRIALS = 20
MAX_LENGTHENING = 10.0


class StepOutcome(NamedTuple):
    accepted: Any
    x: Any  # new incumbent (unchanged if not accepted)
    value: Any
    grad: Any
    objective_evaluations: Any  # int32
    gradient_evaluations: Any  # int32
    trials: Any  # int32
    initial_accepted: Any  # the model's own step (α = 1, shrink = 1) was accepted as is
    wolfe_satisfied: Any
    trial_x: Any  # last evaluated trial (for rejection information)
    trial_value: Any
    next_shrink: Any
    applied_scale: Any  # α · shrink · step of the accepted (else last) trial


def _point(u: Any, base: Any, scale: Any, lo: Any, hi: Any) -> Any:
    u_t = jnp.clip(u - scale * base, -1.0, 1.0)
    return jnp.clip(denormalize_coordinates(u_t, lo, hi), lo, hi)


def _halving(f: Any, grad_f: Any, x: Any, value: Any, grad: Any, base: Any, step: Any,
             shrink: Any, lo: Any, hi: Any) -> StepOutcome:
    u = normalize_coordinates(x, lo, hi)
    scale = (1.0 * shrink) * step
    candidate = _point(u, base, scale, lo, hi)
    candidate_value = f(candidate).astype(value.dtype)
    improved = jnp.isfinite(candidate_value) & (candidate_value < value)
    candidate_grad = jax.lax.cond(improved, lambda c: grad_f(c).astype(grad.dtype),
                                  jnp.zeros_like, candidate)
    accepted = improved & jnp.all(jnp.isfinite(candidate_grad))
    one = jnp.asarray(1, jnp.int32)
    return StepOutcome(
        accepted=accepted,
        x=jnp.where(accepted, candidate, x),
        value=jnp.where(accepted, candidate_value, value),
        grad=jnp.where(accepted, candidate_grad, grad),
        objective_evaluations=one,
        gradient_evaluations=improved.astype(jnp.int32),
        trials=one,
        initial_accepted=accepted & (shrink == 1.0),
        wolfe_satisfied=jnp.asarray(False),
        trial_x=candidate,
        trial_value=candidate_value,
        next_shrink=jnp.where(accepted, jnp.asarray(1.0, value.dtype), 0.5 * shrink),
        applied_scale=scale,
    )


def _armijo(f: Any, grad_f: Any, x: Any, value: Any, grad: Any, base: Any, step: Any,
            shrink: Any, units_left: Any, lo: Any, hi: Any) -> StepOutcome:
    dtype = value.dtype
    u = normalize_coordinates(x, lo, hi)
    g_u = gradient_to_normalized(grad, lo, hi)
    dphi0 = -(shrink * step) * jnp.sum(g_u * base)  # dφ/dα at 0 along the unclipped ray

    def keep(state: dict[str, Any]) -> Any:
        return (~state["found"]) & (state["trials"] < MAX_TRIALS) & (
            units_left - state["trials"] >= 2)

    def body(state: dict[str, Any]) -> dict[str, Any]:
        alpha = state["alpha"]
        scale = (alpha * shrink) * step
        trial = _point(u, base, scale, lo, hi)
        trial_value = f(trial).astype(dtype)
        du = normalize_coordinates(trial, lo, hi) - u
        finite = jnp.isfinite(trial_value)
        ok = finite & (trial_value < value) & (trial_value <= value + C1 * jnp.sum(g_u * du))
        q = trial_value - value - dphi0 * alpha
        alpha_q = -dphi0 * alpha**2 / jnp.where(q > 0, 2.0 * q, 1.0)
        usable = finite & (q > 0) & jnp.isfinite(alpha_q)
        alpha_next = jnp.where(usable, jnp.clip(alpha_q, 0.1 * alpha, 0.5 * alpha), 0.5 * alpha)
        return dict(alpha=jnp.where(ok, alpha, alpha_next), trials=state["trials"] + 1, found=ok,
                    trial=trial, trial_value=trial_value, scale=scale)

    init = dict(alpha=jnp.asarray(1.0, dtype), trials=jnp.asarray(0, jnp.int32),
                found=jnp.asarray(False), trial=x, trial_value=value,
                scale=jnp.asarray(0.0, dtype))
    out = jax.lax.while_loop(keep, body, init)
    candidate_grad = jax.lax.cond(out["found"], lambda c: grad_f(c).astype(grad.dtype),
                                  jnp.zeros_like, out["trial"])
    accepted = out["found"] & jnp.all(jnp.isfinite(candidate_grad))
    return StepOutcome(
        accepted=accepted,
        x=jnp.where(accepted, out["trial"], x),
        value=jnp.where(accepted, out["trial_value"], value),
        grad=jnp.where(accepted, candidate_grad, grad),
        objective_evaluations=out["trials"],
        gradient_evaluations=out["found"].astype(jnp.int32),
        trials=out["trials"],
        initial_accepted=accepted & (out["trials"] == 1) & (shrink == 1.0),
        wolfe_satisfied=jnp.asarray(False),
        trial_x=out["trial"],
        trial_value=out["trial_value"],
        next_shrink=jnp.where(accepted, jnp.asarray(1.0, dtype), 0.5 * shrink),
        applied_scale=out["scale"],
    )


def _wolfe(f: Any, grad_f: Any, x: Any, value: Any, grad: Any, base: Any, step: Any,
           shrink: Any, units_left: Any, lo: Any, hi: Any) -> StepOutcome:
    dtype = value.dtype
    u = normalize_coordinates(x, lo, hi)
    g_u = gradient_to_normalized(grad, lo, hi)
    velocity = -(shrink * step) * base  # u(α) = u + α · velocity
    dphi0 = jnp.sum(g_u * velocity)
    to_bound = jnp.where(velocity > 0, (1.0 - u) / jnp.where(velocity > 0, velocity, 1.0),
                         jnp.where(velocity < 0, (-1.0 - u) / jnp.where(velocity < 0, velocity,
                                                                       -1.0), jnp.inf))
    alpha_max = jnp.minimum(jnp.min(to_bound), MAX_LENGTHENING)
    inf = jnp.asarray(jnp.inf, dtype)

    def evaluate(alpha: Any) -> tuple[Any, Any, Any, Any, Any]:
        trial = jnp.clip(denormalize_coordinates(u + alpha * velocity, lo, hi), lo, hi)
        trial_value = f(trial).astype(dtype)
        trial_grad = grad_f(trial).astype(grad.dtype)
        slope = jnp.sum(gradient_to_normalized(trial_grad, lo, hi) * velocity)
        finite = jnp.isfinite(trial_value) & jnp.all(jnp.isfinite(trial_grad))
        return trial, trial_value, trial_grad, slope, finite

    def keep(s: dict[str, Any]) -> Any:
        return (~s["done"]) & (s["trials"] < MAX_TRIALS) & (units_left - 2 * s["trials"] >= 2)

    def body(s: dict[str, Any]) -> dict[str, Any]:
        alpha = s["alpha"]
        trial, ft, gt, dt, finite = evaluate(alpha)
        better = finite & (ft < s["best_value"])
        best = {k: jnp.where(better, new, s[k]) for k, new in (
            ("best_x", trial), ("best_value", ft), ("best_grad", gt), ("best_alpha", alpha))}
        bad = (
            (~finite)
            | (ft > value + C1 * alpha * dphi0)
            | ((s["trials"] > 0) & (ft >= s["f_lo"]))
        )
        curvature = finite & (jnp.abs(dt) <= -C2 * dphi0)
        satisfied = (~bad) & curvature
        zoom = s["phase"] == 1
        # Bracketing (phase 0).
        to_zoom_hi = (~zoom) & bad
        to_zoom_flip = (~zoom) & (~bad) & (~curvature) & (dt >= 0)
        extend = (~zoom) & (~bad) & (~curvature) & (dt < 0)
        truncated = extend & (alpha >= alpha_max)
        # Zoom (phase 1).
        zoom_bad = zoom & bad
        zoom_move = zoom & (~bad) & (~curvature)
        flip_hi = zoom_move & (dt * (s["a_hi"] - s["a_lo"]) >= 0)
        a_lo = jnp.where(to_zoom_flip | extend | zoom_move, alpha, s["a_lo"])
        f_lo = jnp.where(to_zoom_flip | extend | zoom_move, ft, s["f_lo"])
        d_lo = jnp.where(to_zoom_flip | extend | zoom_move, dt, s["d_lo"])
        a_hi = jnp.where(to_zoom_hi | zoom_bad, alpha,
                         jnp.where(to_zoom_flip | flip_hi, s["a_lo"], s["a_hi"]))
        f_hi = jnp.where(to_zoom_hi | zoom_bad, jnp.where(finite, ft, inf),
                         jnp.where(to_zoom_flip | flip_hi, s["f_lo"], s["f_hi"]))
        phase = jnp.where(to_zoom_hi | to_zoom_flip, 1, s["phase"])
        width = a_hi - a_lo
        denominator = 2.0 * (f_hi - f_lo - d_lo * width)
        alpha_q = a_lo - d_lo * width**2 / jnp.where(denominator > 0, denominator, 1.0)
        low, high = jnp.minimum(a_lo, a_hi), jnp.maximum(a_lo, a_hi)
        margin = 0.1 * jnp.abs(width)
        interpolated = jnp.isfinite(f_hi) & (denominator > 0) & jnp.isfinite(alpha_q) & (
            alpha_q >= low + margin) & (alpha_q <= high - margin)
        alpha_zoom = jnp.where(interpolated, alpha_q, 0.5 * (a_lo + a_hi))
        collapsed = (phase == 1) & (jnp.abs(width) <= 1e-12 * jnp.maximum(1.0, jnp.abs(a_lo)))
        alpha_next = jnp.where(phase == 1, alpha_zoom, jnp.minimum(2.0 * alpha, alpha_max))
        return dict(
            alpha=alpha_next, a_lo=a_lo, f_lo=f_lo, d_lo=d_lo, a_hi=a_hi, f_hi=f_hi,
            phase=phase, trials=s["trials"] + 1, done=satisfied | truncated | collapsed,
            satisfied=satisfied, last_x=trial, last_value=ft, **best,
        )

    init = dict(
        alpha=jnp.minimum(jnp.asarray(1.0, dtype), alpha_max), a_lo=jnp.asarray(0.0, dtype),
        f_lo=value, d_lo=dphi0, a_hi=alpha_max.astype(dtype), f_hi=inf,
        phase=jnp.asarray(0, jnp.int32), trials=jnp.asarray(0, jnp.int32),
        done=jnp.asarray(False), satisfied=jnp.asarray(False), last_x=x, last_value=value,
        best_x=x, best_value=value, best_grad=grad, best_alpha=jnp.asarray(0.0, dtype),
    )
    out = jax.lax.while_loop(keep, body, init)
    accepted = out["best_value"] < value
    trials = out["trials"]
    return StepOutcome(
        accepted=accepted,
        x=out["best_x"],
        value=out["best_value"],
        grad=out["best_grad"],
        objective_evaluations=trials,
        gradient_evaluations=trials,
        trials=trials,
        initial_accepted=accepted & out["satisfied"] & (trials == 1) & (shrink == 1.0)
        & (out["best_alpha"] == 1.0),
        wolfe_satisfied=out["satisfied"],
        trial_x=out["last_x"],
        trial_value=out["last_value"],
        next_shrink=jnp.where(accepted, jnp.asarray(1.0, dtype), 0.5 * shrink),
        applied_scale=jnp.where(accepted, out["best_alpha"], out["alpha"]) * shrink * step,
    )


def accept_step(rule: str, f: Any, grad_f: Any, x: Any, value: Any, grad: Any, base: Any,
                step: Any, shrink: Any, units_left: Any, lo: Any, hi: Any) -> StepOutcome:
    """Apply a step-acceptance rule to the model's proposed displacement.

    The caller guarantees ``units_left >= 2`` (room for one trial and one gradient).
    """
    if rule == "halving":
        return _halving(f, grad_f, x, value, grad, base, step, shrink, lo, hi)
    if rule == "armijo":
        return _armijo(f, grad_f, x, value, grad, base, step, shrink, units_left, lo, hi)
    if rule == "wolfe":
        u = normalize_coordinates(x, lo, hi)
        velocity = -(shrink * step) * base
        slope = jnp.sum(gradient_to_normalized(grad, lo, hi) * velocity)
        room = jnp.min(jnp.where(velocity > 0, 1.0 - u, jnp.where(velocity < 0, u + 1.0, 1.0)))
        usable = jnp.isfinite(slope) & (slope < 0) & (room > 0)
        return jax.lax.cond(
            usable,
            lambda: _wolfe(f, grad_f, x, value, grad, base, step, shrink, units_left, lo, hi),
            lambda: _halving(f, grad_f, x, value, grad, base, step, shrink, lo, hi),
        )
    raise ValueError(f"loop_rule must be one of {LOOP_RULES}")
