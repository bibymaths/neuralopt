"""v0.8-B baselines and metrics (harness only; plan docs/plans/neuralopt-v08b-step-policy.md).

A *trace* records every objective value an arm obtained, in the order of its budget:
``units[k]`` is the cumulative unit count after which ``values[k]`` was available
(objective call = 1 unit, gradient call = 1 unit; a joint value-and-gradient call costs 2
units and its value counts at the second). ``points[k]`` is the evaluated position (NaN if
the solver does not expose it). ``gradient_units`` lists the units at which gradient calls
completed (when known), so calls within a smaller budget can be counted.

CMA-ES (``cma``) and TPE (``optuna``) are harness-only references from the ``benchmark``
extra; NeuralOpt's package never imports them.
"""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuralopt.compiled import preconditioned_solve_batch
from neuralopt.gram import HISTORY_V2_MASKS, GramConfig, init_gram_model
from neuralopt.lbfgs import lbfgs_solve_batch

POLISH_UNITS = 2000


# ---- traces and metrics ----------------------------------------------------------------


def make_trace(units: Any, values: Any, points: Any, objective_calls: int, gradient_calls: int,
               gradient_units: Any = None) -> dict[str, Any]:
    values = np.asarray(values, float)
    return {"units": np.asarray(units, int), "values": np.where(np.isfinite(values), values,
                                                                np.inf),
            "points": np.asarray(points, float), "objective_calls": int(objective_calls),
            "gradient_calls": int(gradient_calls),
            "gradient_units": None if gradient_units is None else np.asarray(gradient_units,
                                                                             int)}


def best_so_far(trace: dict[str, Any], budget: int) -> np.ndarray:
    """``f_best(t)`` for t = 1..budget (index t − 1); +inf before the first value."""
    best = np.full(budget, np.inf)
    for u, v in zip(trace["units"], trace["values"], strict=True):
        if 1 <= u <= budget:
            best[u - 1] = min(best[u - 1], v)
    return np.minimum.accumulate(best)


def clipped_log_regret(fbest: np.ndarray, f_star: float, tau: float, scale: float) -> np.ndarray:
    """``L(t) = log10(min(max(f_best − f*, τ), 10·s) / s)``; 1 before the first value."""
    regret = np.clip(np.asarray(fbest, float) - f_star, tau, 10.0 * scale)
    return np.log10(regret / scale)


def anytime_auc(trace: dict[str, Any], budget: int, f_star: float, tau: float,
                scale: float) -> float:
    return float(np.mean(clipped_log_regret(best_so_far(trace, budget), f_star, tau, scale)))


def best_point(trace: dict[str, Any], budget: int) -> np.ndarray | None:
    within = np.flatnonzero(trace["units"] <= budget)
    if within.size == 0:
        return None
    vals = trace["values"][within]
    k = within[np.flatnonzero(vals == vals.min())[-1]]  # ties: the latest (owned records it)
    point = trace["points"][k]
    if not np.all(np.isfinite(point)):
        raise ValueError("the best point within this budget is not recorded by the solver")
    return point


def calls_within(trace: dict[str, Any], budget: int) -> tuple[int, int]:
    """(objective, gradient) calls completed within ``budget`` (traces with known units)."""
    if trace["gradient_units"] is None:
        raise ValueError("this trace does not record gradient units; use the run totals")
    return (int(np.sum(trace["units"] <= budget)),
            int(np.sum(trace["gradient_units"] <= budget)))


def combine_multistart(traces: list[dict[str, Any]], per_start: int) -> dict[str, Any]:
    """Starts run one after another; start k's units are offset by ``k · per_start``."""
    units, values, points, grads = [], [], [], []
    objective = gradient = 0
    for k, tr in enumerate(traces):
        keep = tr["units"] <= per_start
        units.append(tr["units"][keep] + k * per_start)
        values.append(tr["values"][keep])
        points.append(tr["points"][keep])
        if tr["gradient_units"] is not None:
            g = tr["gradient_units"][tr["gradient_units"] <= per_start]
            grads.append(g + k * per_start)
            gradient += g.size
            objective += int(np.sum(keep))
        else:
            objective += tr["objective_calls"]
            gradient += tr["gradient_calls"]
    known = all(tr["gradient_units"] is not None for tr in traces)
    return make_trace(np.concatenate(units), np.concatenate(values), np.concatenate(points),
                      objective, gradient, np.concatenate(grads) if known else None)


# ---- gradient-free baselines (one unit per objective call) ------------------------------


def _to_u(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def _from_u(u: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + (np.clip(u, -1.0, 1.0) + 1.0) * (hi - lo) / 2.0


def quasi_random(f_batch: Any, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: int,
                 seed: int) -> dict[str, Any]:
    """Scrambled Sobol points in the box; the first evaluation is ``x0``."""
    from scipy.stats import qmc

    m = int(np.ceil(np.log2(max(budget, 2))))
    u = qmc.Sobol(len(x0), scramble=True, seed=seed).random_base2(m)[:budget]
    points = lo + (hi - lo) * u
    points[0] = x0
    values = np.asarray(f_batch(points), float)
    return make_trace(np.arange(1, budget + 1), values, points, budget, 0, np.zeros(0, int))


def cmaes(f_batch: Any, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: int,
          sigma0: float, popsize: int | None, seed: int) -> dict[str, Any]:
    """pycma in box-normalized coordinates (u ∈ [−1, 1]^d), mean at ``x0``; ``sigma0`` is a
    fraction of the u-box width (2). ``x0`` is evaluated first; the last generation is
    truncated at the budget (and then not told). Stopping rules are disabled so the whole
    budget is used."""
    import cma

    points = [np.asarray(x0, float)]
    values = [float(np.asarray(f_batch(np.asarray(x0, float)[None]))[0])]
    options = {"bounds": [-1.0, 1.0], "seed": seed + 1, "verbose": -9,
               "tolfun": 0, "tolx": 0, "tolfunhist": 0, "tolstagnation": 10**9,
               "tolflatfitness": 10**9, "maxfevals": 10**9}
    if popsize is not None:
        options["popsize"] = popsize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = cma.CMAEvolutionStrategy(_to_u(np.asarray(x0, float), lo, hi), 2.0 * sigma0,
                                      options)
        while len(values) < budget:
            ask = [np.asarray(a, float) for a in es.ask()]
            take = ask[: budget - len(values)]
            xs = np.stack([_from_u(a, lo, hi) for a in take])
            vals = np.asarray(f_batch(xs), float)
            points.extend(xs)
            values.extend(vals.tolist())
            if len(take) < len(ask):
                break
            es.tell(ask, [v if np.isfinite(v) else 1e300 for v in vals])
    return make_trace(np.arange(1, len(values) + 1), values, np.stack(points), len(values), 0,
                      np.zeros(0, int))


def tpe(f: Any, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: int, n_startup: int,
        multivariate: bool, seed: int) -> dict[str, Any]:
    """optuna's TPE sampler (in-memory study); the first trial is enqueued at ``x0``."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = len(x0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup,
                                             multivariate=multivariate)
        study = optuna.create_study(sampler=sampler)
        study.enqueue_trial({f"x{j}": float(x0[j]) for j in range(d)})
        points: list[np.ndarray] = []
        values: list[float] = []

        def objective(trial: Any) -> float:
            x = np.array([trial.suggest_float(f"x{j}", float(lo[j]), float(hi[j]))
                          for j in range(d)])
            v = float(f(x))
            points.append(x)
            values.append(v)
            return v if np.isfinite(v) else 1e300

        study.optimize(objective, n_trials=budget)
    return make_trace(np.arange(1, budget + 1), values, np.stack(points), budget, 0,
                      np.zeros(0, int))


# ---- gradient-based baselines -----------------------------------------------------------


def scipy_lbfgsb(value_and_grad: Any, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 budget: int) -> dict[str, Any]:
    """SciPy L-BFGS-B with tolerances that let it use the full budget (ftol = gtol = 0,
    maxfun = budget/2); each call is objective + gradient = 2 units."""
    from scipy.optimize import minimize as scipy_minimize

    points: list[np.ndarray] = []
    values: list[float] = []

    class Exhausted(Exception):
        pass

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        if 2 * (len(values) + 1) > budget:
            raise Exhausted
        v, g = value_and_grad(np.asarray(x, float))
        points.append(np.asarray(x, float).copy())
        values.append(float(v))
        g = np.asarray(g, float)
        return (float(v) if np.isfinite(v) else 1e300), np.where(np.isfinite(g), g, 0.0)

    try:
        scipy_minimize(fun, np.clip(x0, lo, hi), jac=True, method="L-BFGS-B",
                       bounds=list(zip(lo, hi, strict=True)),
                       options={"maxfun": budget // 2, "maxiter": budget, "ftol": 0.0,
                                "gtol": 0.0, "maxcor": 10})
    except Exhausted:
        pass
    n = len(values)
    units = 2 * np.arange(1, n + 1)
    return make_trace(units, values, np.stack(points) if n else np.zeros((0, len(x0))), n, n,
                      units)


def _tile_bounds(lo: np.ndarray, hi: np.ndarray, n: int) -> tuple[Any, Any]:
    return jnp.tile(jnp.asarray(lo), (n, 1)), jnp.tile(jnp.asarray(hi), (n, 1))


_BATCHED: dict[Any, Any] = {}


def batched_objective(objective: Any) -> Any:
    if objective not in _BATCHED:
        _BATCHED[objective] = jax.jit(jax.vmap(objective))
    return _BATCHED[objective]


def owned_lbfgs(objective: Any, contexts: Any, x0s: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                budget: int, return_raw: bool = False) -> Any:
    """NeuralOpt's own L-BFGS (sequential batch strategy), one trace per instance.

    The start's value is available after its objective call (unit 1); an accepted trial's
    value after its objective call, one unit before the acceptance gradient. Rejected
    line-search trials are not exposed by the solver: they are counted in the units but not
    in the trace (their values exceed the current value minus the tiny Armijo margin). Only
    the final point is recorded, so :func:`best_point` is valid at the run's own budget.
    """
    n = x0s.shape[0]
    out = lbfgs_solve_batch(objective, contexts, jnp.asarray(x0s), _tile_bounds(lo, hi, n),
                            budget, strategy="map")
    f0 = np.asarray(batched_objective(objective)(jnp.asarray(x0s), contexts))
    traces = []
    for i in range(n):
        k = int(out["accepted"][i])
        units = [1, *(np.asarray(out["accepted_units"][i][:k]) - 1).tolist()]
        values = [float(f0[i]), *np.asarray(out["accepted_values"][i][:k]).tolist()]
        points = np.full((k + 1, x0s.shape[1]), np.nan)
        points[-1] = np.asarray(out["x"][i])
        traces.append(make_trace(units, values, points, int(out["objective_evaluations"][i]),
                                 int(out["gradient_evaluations"][i])))
    return (traces, out) if return_raw else traces


def bb_config(fallback_length: float) -> GramConfig:
    """The v0.7 reference Gram configuration; with all-zero parameters its loop is BB."""
    return GramConfig(history_length=10, width=32, heads=4, layers=2, **HISTORY_V2_MASKS,
                      fallback_length=fallback_length)


def bb(objective: Any, contexts: Any, x0s: np.ndarray, lo: np.ndarray, hi: np.ndarray,
       budget: int, fallback_length: float, return_raw: bool = False) -> Any:
    """NeuralOpt's compiled-loop BB step (Gram loop with zero parameters, halving rule)."""
    config = bb_config(fallback_length)
    params = {k: jnp.zeros_like(v)
              for k, v in init_gram_model(jax.random.PRNGKey(0), config).items()}
    n = x0s.shape[0]
    out = preconditioned_solve_batch(objective, contexts, jnp.asarray(x0s),
                                     _tile_bounds(lo, hi, n), budget, params=params,
                                     config=config, model_kind="gram")
    traces = []
    for i in range(n):
        iterations = int(out["iteration"][i])
        if np.any(np.asarray(out["trials"][i][:iterations]) != 1):
            raise RuntimeError("BB loop made more than one trial in an iteration")
        units, values, points, grads = [1], [float(out["trajectory"][i][0])], [x0s[i]], [2]
        used = 2
        for j in range(iterations):
            units.append(used + 1)
            values.append(float(out["candidate_values"][i][j]))
            points.append(np.asarray(out["candidates"][i][j]))
            if bool(out["accepted"][i][j]):
                grads.append(used + 2)
                used += 2
            else:
                used += 1
        if used != int(out["used"][i]):
            raise RuntimeError("BB unit reconstruction does not match the loop's count")
        traces.append(make_trace(units, values, np.stack(points), len(units), len(grads),
                                 grads))
    return (traces, out) if return_raw else traces


# ---- measurement-only polish (uncharged) ------------------------------------------------


def polish(objective: Any, contexts: Any, xs: np.ndarray, lo: np.ndarray, hi: np.ndarray,
           tile: int = 1, budget: int = POLISH_UNITS) -> np.ndarray:
    """Owned L-BFGS from each point (gtol 1e-8, ≤ 2,000 units, no restarts); final values.

    ``xs`` holds ``tile`` consecutive blocks, one point per instance in each block.
    """
    ctx = jax.tree_util.tree_map(lambda a: jnp.concatenate([a] * tile), contexts)
    xs = np.clip(np.asarray(xs, float), lo, hi)
    out = lbfgs_solve_batch(objective, ctx, jnp.asarray(xs), _tile_bounds(lo, hi, len(xs)),
                            budget, strategy="map")
    return np.asarray(out["fun"], float)
