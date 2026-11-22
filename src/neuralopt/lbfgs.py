"""NeuralOpt's own L-BFGS: classical (not learned), pure JAX, projected box handling.

v0.8 refinement for every NeuralOpt arm (``docs/plans/neuralopt-v08-amortized-families.md``,
section 3). One ``lax.while_loop`` inside one ``jit``; ``vmap``-able over instances.

Method:
- Limited-memory BFGS two-loop recursion over stored curvature pairs; a pair is stored only
  if ``sᵀy > eps ‖s‖ ‖y‖`` (otherwise it is skipped and counted). Initial scaling is
  ``sᵀy / yᵀy`` of the newest pair, or ``1 / ‖g‖`` before any pair exists.
- Box bounds by projection: the direction is computed on the free set (variables not at a
  bound whose gradient pushes them outward), trial points are ``P_box(x + α d)``, and
  backtracking Armijo is checked along the projection arc against ``gᵀ(x_trial − x)``.
  A non-descent direction is replaced by the scaled projected gradient.

Accounting (the convention of :mod:`neuralopt.compiled`): every trial costs one objective
call; the gradient is computed only at an accepted point; a trial is proposed only while
``used + 2 <= budget``; a nonfinite trial value rejects the trial and is counted. Under
``vmap`` both ``lax.cond`` branches are computed, but counters charge only the branch taken.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

(RUNNING, CONVERGED_GTOL, CONVERGED_FTOL, BUDGET_EXHAUSTED, LINE_SEARCH_FAILED,
 NONFINITE_START, NONFINITE_GRADIENT) = range(7)
STATUS_NAMES = {
    RUNNING: "running",
    CONVERGED_GTOL: "converged_gtol",
    CONVERGED_FTOL: "converged_ftol",
    BUDGET_EXHAUSTED: "budget_exhausted",
    LINE_SEARCH_FAILED: "line_search_failed",
    NONFINITE_START: "nonfinite_start",
    NONFINITE_GRADIENT: "nonfinite_gradient",
}


@dataclass(frozen=True)
class LBFGSConfig:
    memory: int = 10
    gtol: float = 1e-8  # projected-gradient infinity norm
    ftol: float = 0.0  # relative-decrease stop; 0 disables it
    c1: float = 1e-4  # Armijo constant
    shrink: float = 0.5  # backtracking factor
    max_backtracks: int = 40
    curvature_eps: float = 1e-10


DEFAULT_LBFGS_CONFIG = LBFGSConfig()


class _State(NamedTuple):
    x: Any
    f: Any
    g: Any
    S: Any  # (m, d)
    Y: Any  # (m, d)
    rho: Any  # (m,), 0 for empty slots
    valid: Any  # (m,) bool
    head: Any  # next slot to write
    used: Any
    objective_evaluations: Any
    gradient_evaluations: Any
    accepted: Any
    iteration: Any
    status: Any
    skipped_pairs: Any
    nonfinite_trials: Any
    accepted_units: Any  # (budget,) units used after each accepted step
    accepted_values: Any  # (budget,) objective value of each accepted step
    trial_min: Any  # (d,) smallest coordinate of any trial point
    trial_max: Any  # (d,)


def _projected_gradient(x: Any, g: Any, lo: Any, hi: Any) -> Any:
    return jnp.clip(x - g, lo, hi) - x


def _two_loop(q: Any, s: _State, memory: int) -> Any:
    """``H q`` for the stored pairs (empty slots have ``rho = 0`` and are no-ops)."""
    order = (s.head - 1 - jnp.arange(memory)) % memory  # newest first

    def first(j: int, carry: tuple[Any, Any]) -> tuple[Any, Any]:
        q, alpha = carry
        i = order[j]
        a = s.rho[i] * jnp.dot(s.S[i], q)
        return q - a * s.Y[i], alpha.at[i].set(a)

    q, alpha = jax.lax.fori_loop(0, memory, first, (q, jnp.zeros((memory,), q.dtype)))
    newest = order[0]
    yy = jnp.dot(s.Y[newest], s.Y[newest])
    gamma = jnp.where(
        s.valid[newest],
        jnp.dot(s.S[newest], s.Y[newest]) / jnp.where(yy > 0, yy, 1.0),
        1.0 / jnp.maximum(jnp.linalg.norm(q), 1e-300),
    )
    r = gamma * q

    def second(j: int, r: Any) -> Any:
        i = order[memory - 1 - j]  # oldest first
        b = s.rho[i] * jnp.dot(s.Y[i], r)
        return r + s.S[i] * (alpha[i] - b)

    return jax.lax.fori_loop(0, memory, second, r), gamma


def _solve(objective: Any, parametric: bool, budget: int, config: LBFGSConfig, theta: Any,
           x0: Any, lo: Any, hi: Any) -> tuple[_State, Any]:
    dtype = lo.dtype
    d = x0.shape[-1]
    m = config.memory

    def f(x: Any) -> Any:
        return (objective(x, theta) if parametric else objective(x)).astype(dtype)

    grad_f = jax.grad(f)
    x0 = jnp.clip(x0.astype(dtype), lo, hi)
    f0, g0 = f(x0), grad_f(x0).astype(dtype)
    ok = jnp.isfinite(f0) & jnp.all(jnp.isfinite(g0))
    zero = jnp.asarray(0, jnp.int32)
    state = _State(
        x=x0, f=f0, g=g0,
        S=jnp.zeros((m, d), dtype), Y=jnp.zeros((m, d), dtype), rho=jnp.zeros((m,), dtype),
        valid=jnp.zeros((m,), bool), head=zero,
        used=jnp.asarray(2, jnp.int32), objective_evaluations=jnp.asarray(1, jnp.int32),
        gradient_evaluations=jnp.asarray(1, jnp.int32), accepted=zero, iteration=zero,
        status=jnp.where(ok, RUNNING, NONFINITE_START).astype(jnp.int32),
        skipped_pairs=zero, nonfinite_trials=zero,
        accepted_units=jnp.zeros((budget,), jnp.int32),
        accepted_values=jnp.full((budget,), jnp.nan, dtype),
        trial_min=x0, trial_max=x0,
    )

    def pg_norm(s: _State) -> Any:
        return jnp.max(jnp.abs(_projected_gradient(s.x, s.g, lo, hi)))

    def keep_going(s: _State) -> Any:
        return (s.status == RUNNING) & (s.used + 2 <= budget) & (pg_norm(s) > config.gtol)

    def step(s: _State) -> _State:
        free = ~(((s.x <= lo) & (s.g > 0)) | ((s.x >= hi) & (s.g < 0)))
        g_free = jnp.where(free, s.g, 0.0)
        hq, gamma = _two_loop(g_free, s, m)
        direction = jnp.where(free, -hq, 0.0)
        descent = jnp.dot(s.g, direction)
        fallback = -gamma * g_free
        direction = jnp.where(jnp.isfinite(descent) & (descent < 0), direction, fallback)

        def search_more(c: tuple) -> Any:
            alpha, k, used, n_obj, bad, found, xt, ft, tmin, tmax = c
            return (~found) & (k < config.max_backtracks) & (used + 2 <= budget)

        def trial(c: tuple) -> tuple:
            alpha, k, used, n_obj, bad, found, xt, ft, tmin, tmax = c
            xt = jnp.clip(s.x + alpha * direction, lo, hi)
            ft = f(xt)
            finite = jnp.isfinite(ft)
            predicted = jnp.dot(s.g, xt - s.x)
            armijo = finite & (predicted < 0) & (ft <= s.f + config.c1 * predicted)
            return (alpha * config.shrink, k + 1, used + 1, n_obj + 1,
                    bad + (~finite).astype(jnp.int32), armijo, xt, ft,
                    jnp.minimum(tmin, xt), jnp.maximum(tmax, xt))

        start = (jnp.asarray(1.0, dtype), zero, s.used, s.objective_evaluations,
                 s.nonfinite_trials, jnp.asarray(False), s.x, s.f, s.trial_min, s.trial_max)
        _, _, used, n_obj, bad, found, xt, ft, tmin, tmax = jax.lax.while_loop(
            search_more, trial, start)
        s = s._replace(used=used, objective_evaluations=n_obj, nonfinite_trials=bad,
                       trial_min=tmin, trial_max=tmax, iteration=s.iteration + 1)

        def accept(s: _State) -> _State:
            gt = grad_f(xt).astype(dtype)
            used = s.used + 1
            good = jnp.all(jnp.isfinite(gt))
            sv, yv = xt - s.x, gt - s.g
            sy = jnp.dot(sv, yv)
            store = good & (sy > config.curvature_eps * jnp.linalg.norm(sv)
                            * jnp.linalg.norm(yv))
            i = s.head
            rel = (s.f - ft) / jnp.maximum(jnp.maximum(jnp.abs(s.f), jnp.abs(ft)), 1.0)
            status = jnp.where(
                ~good, NONFINITE_GRADIENT,
                jnp.where((config.ftol > 0) & (rel <= config.ftol), CONVERGED_FTOL, RUNNING),
            ).astype(jnp.int32)
            return s._replace(
                x=jnp.where(good, xt, s.x), f=jnp.where(good, ft, s.f),
                g=jnp.where(good, gt, s.g),
                S=jnp.where(store, s.S.at[i].set(sv), s.S),
                Y=jnp.where(store, s.Y.at[i].set(yv), s.Y),
                rho=jnp.where(store, s.rho.at[i].set(1.0 / jnp.where(store, sy, 1.0)), s.rho),
                valid=jnp.where(store, s.valid.at[i].set(True), s.valid),
                head=jnp.where(store, (i + 1) % m, i).astype(jnp.int32),
                used=used, gradient_evaluations=s.gradient_evaluations + 1,
                accepted=s.accepted + good.astype(jnp.int32),
                skipped_pairs=s.skipped_pairs + (good & ~store).astype(jnp.int32),
                status=status,
                accepted_units=jnp.where(good, s.accepted_units.at[s.accepted].set(used),
                                         s.accepted_units),
                accepted_values=jnp.where(good, s.accepted_values.at[s.accepted].set(ft),
                                          s.accepted_values),
            )

        def reject(s: _State) -> _State:
            out_of_budget = s.used + 2 > budget
            return s._replace(status=jnp.where(out_of_budget, BUDGET_EXHAUSTED,
                                               LINE_SEARCH_FAILED).astype(jnp.int32))

        return jax.lax.cond(found, accept, reject, s)

    final = jax.lax.while_loop(keep_going, step, state)
    norm = pg_norm(final)
    status = jnp.where(
        final.status == RUNNING,
        jnp.where(norm <= config.gtol, CONVERGED_GTOL, BUDGET_EXHAUSTED),
        final.status,
    ).astype(jnp.int32)
    return final._replace(status=status), norm


_STATIC = ("objective", "parametric", "budget", "config")
_solve_jit = jax.jit(_solve, static_argnames=_STATIC)


def _batched(objective: Any, budget: int, config: LBFGSConfig, strategy: str, thetas: Any,
             x0: Any, lo: Any, hi: Any) -> tuple[_State, Any]:
    solve = partial(_solve, objective, True, budget, config)
    if strategy == "map":  # sequential inside one program: each lax.cond runs one branch
        return jax.lax.map(lambda args: solve(*args), (thetas, x0, lo, hi))
    return jax.vmap(solve)(thetas, x0, lo, hi)


_batched_jit = jax.jit(_batched, static_argnames=("objective", "budget", "config", "strategy"))
BATCH_STRATEGIES = ("vmap", "map")


def _to_host(state: _State, norm: Any) -> dict[str, Any]:
    host = jax.device_get(state._asdict())
    host["fun"] = host.pop("f")
    host["grad"] = host.pop("g")
    host["pair_valid"] = host.pop("valid")
    host["projected_gradient_norm"] = jax.device_get(norm)
    # Single solves: 0-d arrays become Python scalars (batched results keep their axis).
    return {k: v.item() if getattr(v, "ndim", None) == 0 else v for k, v in host.items()}


def lbfgs_solve(objective: Any, x0: Any, bounds: tuple[Any, Any], budget: int, *,
                theta: Any = None, config: LBFGSConfig = DEFAULT_LBFGS_CONFIG) -> dict[str, Any]:
    """Minimize ``objective(x)`` (or ``objective(x, theta)``) on a box with ``budget`` units.

    ``objective`` is static: reuse the same function object to reuse the compilation.
    Returns host arrays; ``status`` is a key of :data:`STATUS_NAMES`.
    """
    if budget < 2:
        raise ValueError("L-BFGS needs a budget of at least 2 (objective + gradient)")
    lo, hi = (jnp.asarray(b) for b in bounds)
    state, norm = _solve_jit(objective=objective, parametric=theta is not None, budget=budget,
                             config=config, theta=theta, x0=jnp.asarray(x0, lo.dtype), lo=lo,
                             hi=hi)
    return _to_host(state, norm)


def lbfgs_solve_batch(objective: Any, thetas: Any, x0: Any, bounds: tuple[Any, Any],
                      budget: int, *, config: LBFGSConfig = DEFAULT_LBFGS_CONFIG,
                      strategy: str = "vmap") -> dict[str, Any]:
    """:func:`lbfgs_solve` over a leading task axis of ``thetas``, ``x0`` and the bounds.

    ``strategy="vmap"`` vectorizes (both ``lax.cond`` branches are computed, so a gradient
    is computed on every trial, though only taken branches are charged); ``"map"`` runs
    the instances sequentially in one compiled program (one branch per step), which is
    cheaper when gradients cost much more than objectives. Results are identical.
    """
    if strategy not in BATCH_STRATEGIES:
        raise ValueError(f"strategy must be one of {BATCH_STRATEGIES}")
    if budget < 2:
        raise ValueError("L-BFGS needs a budget of at least 2 (objective + gradient)")
    lo, hi = (jnp.asarray(b) for b in bounds)
    state, norm = _batched_jit(objective=objective, budget=budget, config=config,
                               strategy=strategy, thetas=thetas,
                               x0=jnp.asarray(x0, lo.dtype), lo=lo, hi=hi)
    return _to_host(state, norm)
