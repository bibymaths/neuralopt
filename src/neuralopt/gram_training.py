"""Matched training unroll for the Gram-space model (v0.7 Track B).

The training loop *is* the inference loop: the compiled engine's Gram branch with the
frozen ``"halving"`` step-acceptance rule (``neuralopt.loop_rules.accept_step``), the same
budget accounting (start = 2 units; a trial = 1 unit, plus 1 if it improved) and the
same stopping condition (``used + 2 <= budget`` and a nonzero gradient), run under
``lax.scan`` for ``budget - 3`` steps (the most iterations the loop can run), with
inactive steps masked. Forward values equal the compiled engine (tested to 1e-10).

The history buffer and the token inputs are treated as observed data
(``stop_gradient``); the metric acts on the live current gradient, so the
meta-gradient flows through the current position and ``∇f`` (second order).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .features import gradient_to_normalized, normalize_coordinates
from .gram import (
    GramConfig,
    RejectionInfo,
    apply_gram_model,
    apply_inverse_metric,
    gram_basis,
    gram_tokens,
    tau_scale,
)
from .loop_rules import accept_step
from .trajectory import empty_history, push_history, record_rejection


@dataclass(frozen=True)
class GramUnrollConfig:
    budget: int = 100
    horizons: tuple[int, ...] = (25, 50, 100)
    candidate_weight: float = 0.1
    regret_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.budget < 3:
            raise ValueError("budget must allow at least one iteration (>= 3)")
        if not self.horizons or any(h < 2 or h > self.budget for h in self.horizons):
            raise ValueError("horizons must lie in [2, budget] objective+gradient units")
        if self.candidate_weight < 0 or self.regret_floor <= 0:
            raise ValueError("candidate_weight must be nonnegative and regret_floor positive")

    @property
    def steps(self) -> int:
        """Most iterations at this budget: the start costs 2 units and the loop runs while
        used + 2 <= budget with at least 1 unit per iteration."""
        return self.budget - 3


def gram_unroll(params: dict[str, Any], objective: Any, bounds: tuple[Any, Any], key: Any,
                config: GramConfig, unroll: GramUnrollConfig = GramUnrollConfig(),
                x0: Any | None = None) -> dict[str, Any]:
    lo, hi = bounds
    budget = unroll.budget
    if x0 is None:
        x0 = jax.random.uniform(key, lo.shape, dtype=lo.dtype, minval=lo, maxval=hi)
    dtype = lo.dtype
    grad_f = jax.grad(objective)
    value0 = objective(x0).astype(dtype)
    grad0 = grad_f(x0).astype(dtype)
    history = push_history(empty_history(config.history_length, x0.shape[-1], dtype),
                           normalize_coordinates(x0, lo, hi),
                           gradient_to_normalized(grad0, lo, hi), value0)
    zero = jnp.asarray(0.0, dtype)
    carry0 = dict(x=x0, value=value0, grad=grad0, history=history,
                  shrink=jnp.asarray(1.0, dtype), used=jnp.asarray(2, jnp.int32),
                  rejected_length=zero, rejected_ratio=zero, rejected_has=jnp.asarray(False))
    stop = jax.lax.stop_gradient

    def step(s: dict[str, Any], _: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        g_u = gradient_to_normalized(s["grad"], lo, hi)
        active = (s["used"] + 2 <= budget) & (jnp.linalg.norm(g_u) != 0)
        observed = stop(s["history"])
        g_obs = stop(g_u)
        rejection = RejectionInfo(observed.rejections, stop(s["rejected_length"]),
                                  stop(s["rejected_ratio"]), s["rejected_has"])
        basis = gram_basis(observed, config)
        tau = tau_scale(basis, g_obs, config.fallback_length)
        tokens = gram_tokens(observed, g_obs, rejection, 1 - s["used"].astype(dtype) / budget,
                             config)
        out = apply_gram_model(params, tokens, basis.mask, config)
        base = apply_inverse_metric(g_u, basis, out["A"], out["C"], out["delta"], tau,
                                    config.eps_rel).astype(dtype)
        result = accept_step("halving", objective, grad_f, s["x"], s["value"], s["grad"], base,
                             out["sigma"].astype(dtype), s["shrink"], budget - s["used"], lo, hi)
        accepted = result.accepted
        pushed = push_history(s["history"], normalize_coordinates(result.x, lo, hi),
                              gradient_to_normalized(result.grad, lo, hi), result.value)
        history = jax.tree_util.tree_map(lambda a, b: jnp.where(accepted, a, b), pushed,
                                         record_rejection(s["history"]))
        u = normalize_coordinates(s["x"], lo, hi)
        step_u = normalize_coordinates(result.trial_x, lo, hi) - u
        predicted = jnp.sum(g_u * step_u)
        ratio = jnp.where(
            jnp.isfinite(result.trial_value) & (predicted != 0),
            (result.trial_value - s["value"]) / jnp.where(predicted != 0, predicted, 1.0),
            10.0,
        )
        new = dict(
            x=result.x, value=result.value, grad=result.grad, history=history,
            shrink=result.next_shrink.astype(dtype),
            used=s["used"] + result.objective_evaluations + result.gradient_evaluations,
            rejected_length=jnp.where(accepted, 0.0, jnp.linalg.norm(step_u)).astype(dtype),
            rejected_ratio=jnp.where(accepted, 0.0, jnp.clip(ratio, -10.0, 10.0)).astype(dtype),
            rejected_has=~accepted,
        )
        carry = jax.tree_util.tree_map(lambda n, o: jnp.where(active, n, o), new, s)
        trial = jnp.where(jnp.isfinite(result.trial_value), result.trial_value, s["value"])
        return carry, dict(values=carry["value"], used=carry["used"], active=active,
                           accepted=active & accepted,
                           trial_values=jnp.where(active, trial, carry["value"]))

    final, outputs = jax.lax.scan(step, carry0, None, length=unroll.steps)
    outputs["initial_value"] = value0
    outputs["final_history"] = final["history"]
    return outputs


def gram_regret_loss(params: dict[str, Any], objective: Any, bounds: tuple[Any, Any],
                     optimum: Any, key: Any, config: GramConfig,
                     unroll: GramUnrollConfig = GramUnrollConfig()) -> Any:
    """Mean over unit horizons of log incumbent regret + weighted candidate term."""
    rolled = gram_unroll(params, objective, bounds, key, config, unroll)
    f0 = rolled["initial_value"]
    optimum = jnp.asarray(optimum, dtype=f0.dtype)
    scale = jnp.abs(f0 - optimum) + 1e-12

    def log_regret(values: Any) -> Any:
        return jnp.log(jnp.maximum(values - optimum, 0.0) / scale + unroll.regret_floor)

    within = [rolled["active"] & (rolled["used"] <= h) for h in unroll.horizons]
    incumbents = jnp.stack([
        jnp.minimum(f0, jnp.min(jnp.where(mask, rolled["values"], jnp.inf))) for mask in within
    ])
    active = rolled["active"].astype(f0.dtype)
    candidate = jnp.sum(active * log_regret(rolled["trial_values"])) / jnp.maximum(
        jnp.sum(active), 1.0)
    return jnp.mean(log_regret(incumbents)) + unroll.candidate_weight * candidate
