"""v0.7 loop-rule ablation with the BFGS oracle (harness only; no training).

AMENDED CRITERION (stated by the user before any rerun, 2026-11-22): per task, log10
regret below -12 counts as -12 (regrets below 1e-12 are equal); an arm passes iff its
floored median log10 regret is within one decade of SciPy L-BFGS-B's floored median at
both kappa = 100 and kappa = 1000. The original gate 2 stays FAILED as written.

Arms (same gate tasks as scripts/run_gram_gates_v0_7.py: rotated kappa 100/1000,
d = 7/10/16, 4 per d, seeds 6000+, float64, budget 100 objective+gradient units):
oracle-BFGS (K=8) under "halving", "armijo", "wolfe"; oracle K=10 under "halving";
tau-only (A = C = 0, delta = 0, sigma = 1) and the v0.6 fixed step (0.03 x normalized
gradient) under all three rules; SciPy L-BFGS-B with maxcor 10 (default) and 8.

Every arm steps through the package's ``neuralopt.loop_rules.accept_step``; only the
oracle *direction* lives here. The harness outer loop is checked against the package's
compiled engine (fixed geometry) under each rule before any number is reported.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ["JAX_ENABLE_X64"] = "1"

import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from neuralopt import Problem, minimize  # noqa: E402
from neuralopt.features import gradient_to_normalized, normalize_coordinates  # noqa: E402
from neuralopt.gram import GramConfig, apply_inverse_metric, gram_basis, tau_scale  # noqa: E402
from neuralopt.loop_rules import LOOP_RULES, accept_step  # noqa: E402
from neuralopt.task_families import make_task, quadratic_objective  # noqa: E402
from neuralopt.trajectory import empty_history, push_history, record_rejection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gram_gates_v0_7 import oracle_coefficients  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "loop-rule-ablation-v0.7.0.json"
BUDGET, DIMS, PER_DIM, CONDITIONS = 100, (7, 10, 16), 4, (100.0, 1000.0)
FIXED_STEP, FLOOR, AMENDED_FLOOR = 0.03, 1e-16, -12.0


def task_list() -> list[tuple[float, int, int, Any]]:
    return [
        (c, d, r, make_task("rotated", c, d, 6000 + 100 * int(np.log10(c)) + 10 * d + r))
        for c in CONDITIONS for d in DIMS for r in range(PER_DIM)
    ]


def make_loop(source: str, rule: str, pairs: int, budget: int = BUDGET) -> Any:
    config = GramConfig(history_length=pairs)

    @jax.jit
    def run(theta: Any, x0: Any, lo: Any, hi: Any) -> dict[str, Any]:
        f = lambda x: quadratic_objective(x, theta)  # noqa: E731
        grad_f = jax.grad(f)
        value, grad = f(x0), grad_f(x0)
        hist = push_history(empty_history(pairs, x0.shape[0], x0.dtype),
                            normalize_coordinates(x0, lo, hi),
                            gradient_to_normalized(grad, lo, hi), value)
        i32 = jnp.int32
        state = dict(x=x0, value=value, grad=grad, hist=hist, shrink=jnp.asarray(1.0),
                     used=jnp.asarray(2, i32), objective=jnp.asarray(1, i32),
                     gradient=jnp.asarray(1, i32), iterations=jnp.asarray(0, i32),
                     accepted=jnp.asarray(0, i32), trials=jnp.asarray(0, i32),
                     initial=jnp.asarray(0, i32),
                     trajectory=jnp.full((budget,), jnp.nan).at[0].set(value))

        def keep_going(s: dict[str, Any]) -> Any:
            g_u = gradient_to_normalized(s["grad"], lo, hi)
            return (s["used"] + 2 <= budget) & (jnp.linalg.norm(g_u) != 0)

        def step(s: dict[str, Any]) -> dict[str, Any]:
            g_u = gradient_to_normalized(s["grad"], lo, hi)
            if source == "fixed":  # the package's fixed geometry: base = ĝ, step = L
                base, length = g_u / jnp.linalg.norm(g_u), jnp.asarray(FIXED_STEP)
            else:
                basis = gram_basis(s["hist"])
                tau = tau_scale(basis, g_u, config.fallback_length)
                zeros = jnp.zeros((pairs, pairs))
                if source == "oracle":
                    A, C, delta, _ = oracle_coefficients(basis, g_u, tau)
                else:  # tau-only
                    A, C, delta = zeros, zeros, jnp.asarray(0.0)
                base = apply_inverse_metric(g_u, basis, A, C, delta, tau, config.eps_rel)
                length = jnp.asarray(1.0)
            out = accept_step(rule, f, grad_f, s["x"], s["value"], s["grad"], base, length,
                              s["shrink"], budget - s["used"], lo, hi)
            pushed = push_history(s["hist"], normalize_coordinates(out.x, lo, hi),
                                  gradient_to_normalized(out.grad, lo, hi), out.value)
            hist = jax.tree_util.tree_map(lambda a, b: jnp.where(out.accepted, a, b), pushed,
                                          record_rejection(s["hist"]))
            i = s["iterations"]
            return dict(
                x=out.x, value=out.value, grad=out.grad, hist=hist,
                shrink=out.next_shrink.astype(x0.dtype),
                used=s["used"] + out.objective_evaluations + out.gradient_evaluations,
                objective=s["objective"] + out.objective_evaluations,
                gradient=s["gradient"] + out.gradient_evaluations, iterations=i + 1,
                accepted=s["accepted"] + out.accepted.astype(i32),
                trials=s["trials"] + out.trials,
                initial=s["initial"] + out.initial_accepted.astype(i32),
                trajectory=s["trajectory"].at[i + 1].set(out.value),
            )

        return jax.lax.while_loop(keep_going, step, state)

    return run


def lbfgsb(task: Any, maxcor: int, budget: int = BUDGET) -> tuple[float, int]:
    from scipy.optimize import minimize as scipy_minimize

    calls, best = 0, np.inf

    class Exhausted(Exception):
        pass

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls, best
        if calls >= budget // 2:  # every call, line-search trials included, costs 2 units
            raise Exhausted
        calls += 1
        v, g = _VG(x, task.theta)
        best = min(best, float(v))
        return float(v), np.asarray(g, dtype=float)

    try:
        scipy_minimize(fun, np.asarray(task.x0), jac=True, method="L-BFGS-B",
                       bounds=list(zip(np.asarray(task.lo), np.asarray(task.hi), strict=True)),
                       options={"maxfun": budget // 2, "maxiter": budget, "ftol": 0.0,
                                "gtol": 0.0, "maxcor": maxcor})
    except Exhausted:
        pass
    return best, 2 * calls


_VG = jax.jit(jax.value_and_grad(quadratic_objective))


def parity_checks() -> dict[str, bool]:
    """Harness outer loop == package compiled engine (fixed geometry), every rule."""
    task = make_task("rotated", 100.0, 10, 5998)
    problem = Problem(lambda x: quadratic_objective(x, task.theta), (task.lo, task.hi), 10,
                      x0=np.asarray(task.x0), dtype=jnp.float64)
    out = {}
    for rule in LOOP_RULES:
        ours = make_loop("fixed", rule, 8, 60)(task.theta, task.x0, task.lo, task.hi)
        package = minimize(problem, max_evaluations=60, candidate_method="preconditioned",
                           hamiltonian_metric="fixed", fixed_metric_diag=np.ones(10),
                           preconditioned_step=FIXED_STEP, preconditioned_engine="compiled",
                           loop_rule=rule)
        n = int(ours["iterations"]) + 1
        np.testing.assert_allclose(np.asarray(ours["trajectory"][:n]), package.trajectory,
                                   rtol=1e-10)
        assert int(ours["used"]) == package.evaluations
        out[rule] = True
    return out


def summarize(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        sel = [r for r in rows if r["condition"] == condition]
        ref = float(np.median([max(r["lbfgsb-m10"]["log10_regret"], AMENDED_FLOOR)
                               for r in sel]))
        block = {}
        for arm in arms:
            values = [r[arm]["log10_regret"] for r in sel]
            floored = float(np.median([max(v, AMENDED_FLOOR) for v in values]))
            entry = {
                "median_log10_regret": float(np.median(values)),
                "median_log10_regret_floored_-12": floored,
                "reached_1e-6": f"{sum(v <= -6 for v in values)}/{len(values)}",
                "mean_units": float(np.mean([r[arm]["units"] for r in sel])),
                "amended_pass_vs_lbfgsb_m10": floored <= ref + 1.0,
            }
            if "iterations" in sel[0][arm]:
                iterations = sum(r[arm]["iterations"] for r in sel)
                entry["accepted_fraction_of_iterations"] = sum(
                    r[arm]["accepted"] for r in sel) / max(iterations, 1)
                entry["accepted_fraction_of_trials"] = sum(
                    r[arm]["accepted"] for r in sel) / max(sum(r[arm]["trials"] for r in sel), 1)
                entry["initial_step_accepted_unchanged"] = sum(
                    r[arm]["initial"] for r in sel) / max(iterations, 1)
                entry["mean_trials_per_iteration"] = sum(r[arm]["trials"] for r in sel) / max(
                    iterations, 1)
            block[arm] = entry
        summary[f"kappa_{int(condition)}"] = block
    return summary


def main() -> None:
    assert jax.config.jax_enable_x64
    parity = parity_checks()
    loop_arms = {f"{source}-K{pairs}-{rule}": (source, rule, pairs)
                 for source in ("oracle", "tau-only", "fixed") for rule in LOOP_RULES
                 for pairs in (8,)}
    loop_arms["oracle-K10-halving"] = ("oracle", "halving", 10)
    loops = {name: make_loop(*spec) for name, spec in loop_arms.items()}
    rows = []
    for condition, dim, rep, task in task_list():
        f0 = float(quadratic_objective(task.x0, task.theta))
        row: dict[str, Any] = {"condition": condition, "dimension": dim, "replicate": rep}
        for name, loop in loops.items():
            out = loop(task.theta, task.x0, task.lo, task.hi)
            row[name] = {
                "log10_regret": float(np.log10(max(float(out["value"]), 0.0) / f0 + FLOOR)),
                "units": int(out["used"]),
                "iterations": int(out["iterations"]),
                "accepted": int(out["accepted"]),
                "trials": int(out["trials"]),
                "initial": int(out["initial"]),
            }
        for maxcor in (10, 8):
            best, units = lbfgsb(task, maxcor)
            row[f"lbfgsb-m{maxcor}"] = {
                "log10_regret": float(np.log10(max(best, 0.0) / f0 + FLOOR)), "units": units}
        rows.append(row)
    arms = [*loop_arms, "lbfgsb-m10", "lbfgsb-m8"]
    output = {
        "schema": "neuralopt-loop-rule-ablation/v1",
        "amended_criterion": "per-task log10 regret floored at -12; pass iff floored median "
        "<= SciPy L-BFGS-B (maxcor 10) floored median + 1.0 at kappa 100 and 1000",
        "environment": {"jax": jax.__version__, "python": platform.python_version(),
                        "platform": platform.platform(), "float": "float64"},
        "harness_parity_with_package_engine": parity,
        "summary": summarize(rows, arms),
        "rows": rows,
    }
    RESULTS.write_text(json.dumps(output, indent=1))
    print(json.dumps({"parity": parity, "summary": output["summary"]}, indent=1))


if __name__ == "__main__":
    main()
