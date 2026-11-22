"""v0.7 Track B gate 2 (capacity, no training) and the tf-h8 anisotropy check.

PRE-STATED CRITERION (written before the first run, 2026-11-22):
Gate 2 passes iff, for both rotated kappa = 100 and kappa = 1000 (d in {7, 10, 16},
4 tasks per d, gate-only seeds 6000+, budget 100 objective+gradient units, float64),
the median log10 normalized regret of the oracle-BFGS arm in the shared loop is within
one decade of SciPy L-BFGS-B on the same tasks (oracle <= L-BFGS-B + 1.0). Otherwise
STOP: the loop rule, not the model, is the limit.

The oracle sets A, C, delta to the BFGS values (scripts/gram_reference.py) and
sigma = 1. It is a harness-only reference: the plan forbids a BFGS inference fallback,
so this code never enters ``src/neuralopt``. The harness loop is checked to reproduce
the package's compiled Gram loop exactly before any oracle number is reported.

Usage: ``uv run python scripts/run_gram_gates_v0_7.py``
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ["JAX_ENABLE_X64"] = "1"

import json  # noqa: E402
import platform  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from neuralopt import Problem, minimize  # noqa: E402
from neuralopt.features import (  # noqa: E402
    denormalize_coordinates,
    gradient_to_normalized,
    normalize_coordinates,
)
from neuralopt.gram import (  # noqa: E402
    GramConfig,
    RejectionInfo,
    apply_inverse_metric,
    dense_inverse_metric,
    gram_basis,
    gram_direction,
    init_gram_model,
    tau_scale,
)
from neuralopt.task_families import make_task, quadratic_objective  # noqa: E402
from neuralopt.trajectory import empty_history, push_history, record_rejection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gram-gates-v0.7.0.json"
BUDGET, DIMS, PER_DIM, CONDITIONS = 100, (7, 10, 16), 4, (100.0, 1000.0)
CONFIG = GramConfig(history_length=8, width=32, heads=4, layers=2)
FLOOR = 1e-16


def oracle_coefficients(basis: Any, g_u: Any, tau: Any) -> tuple[Any, Any, Any, Any]:
    """BFGS (A, C, delta) for the valid pairs; masked pairs are decoupled by an identity
    block in R (their rows/columns are zeroed again by the metric's mask)."""
    m = basis.mask
    S = basis.S_hat * basis.s_norm[None, :]
    Y = basis.Y_hat * basis.y_norm[None, :]
    sty = S.T @ Y
    R = jnp.triu(sty) + jnp.diag(jnp.where(m, 0.0, 1.0))
    D = jnp.where(m, jnp.diag(sty), 1.0)
    positive = jnp.all(jnp.where(m, D > 0, True))
    r_inv_t = jnp.linalg.inv(R).T
    A_hat = basis.s_norm[:, None] * r_inv_t * basis.y_norm[None, :]
    root_d = jnp.sqrt(jnp.maximum(D, 0.0))
    C_hat = basis.s_norm[:, None] * r_inv_t * root_d[None, :] / jnp.sqrt(tau)
    newest = jnp.argmax(jnp.where(m, jnp.arange(m.shape[0]), -1))
    yy = jnp.sum(Y[:, newest] ** 2)
    usable = jnp.any(m) & (basis.curvature[newest] > 0) & positive
    gamma = jnp.where(usable, basis.curvature[newest] / jnp.where(yy > 0, yy, 1.0), tau)
    zeros = jnp.zeros_like(A_hat)
    return (jnp.where(usable, A_hat, zeros), jnp.where(usable, C_hat, zeros),
            jnp.where(usable, jnp.log(gamma / tau), 0.0), positive)


def make_loop(source: str, budget: int, config: GramConfig) -> Any:
    """Harness copy of the compiled Gram loop with coefficients from ``source``
    ("oracle", "model" or "tau-only": A = C = 0, delta = 0, sigma = 1)."""

    @jax.jit
    def run(theta: Any, x0: Any, lo: Any, hi: Any, params: Any) -> dict[str, Any]:
        f = lambda x: quadratic_objective(x, theta)  # noqa: E731
        grad_f = jax.grad(f)
        value, grad = f(x0), grad_f(x0)
        hist = push_history(empty_history(config.history_length, x0.shape[0], x0.dtype),
                            normalize_coordinates(x0, lo, hi),
                            gradient_to_normalized(grad, lo, hi), value)
        int32 = jnp.int32
        state = dict(x=x0, value=value, grad=grad, hist=hist, shrink=jnp.asarray(1.0),
                     used=jnp.asarray(2, int32), accepted=jnp.asarray(0, int32),
                     rejected=jnp.asarray(0, int32), iteration=jnp.asarray(0, int32),
                     rl=jnp.asarray(0.0), rr=jnp.asarray(0.0),
                     rh=jnp.asarray(False),
                     trajectory=jnp.full((budget,), jnp.nan).at[0].set(value),
                     max_abs_a=jnp.asarray(0.0), max_abs_c=jnp.asarray(0.0),
                     max_abs_delta=jnp.asarray(0.0),
                     nonpositive=jnp.asarray(0, int32), max_condition=jnp.asarray(1.0))

        def keep_going(s: dict[str, Any]) -> Any:
            g_u = gradient_to_normalized(s["grad"], lo, hi)
            return (s["used"] + 2 <= budget) & (jnp.linalg.norm(g_u) != 0)

        def step(s: dict[str, Any]) -> dict[str, Any]:
            g_u = gradient_to_normalized(s["grad"], lo, hi)
            u = normalize_coordinates(s["x"], lo, hi)
            basis = gram_basis(s["hist"])
            tau = tau_scale(basis, g_u, config.fallback_length)
            positive = jnp.asarray(True)
            if source == "model":
                rejection = RejectionInfo(s["hist"].rejections, s["rl"], s["rr"], s["rh"])
                # Same as the package: cast the int32 counter to float64 before dividing.
                fraction = 1 - s["used"].astype(x0.dtype) / budget
                direction, _, _ = gram_direction(params, config, s["hist"], g_u, rejection,
                                                 fraction)
                A = C = jnp.zeros((config.history_length,) * 2)
                delta = jnp.asarray(0.0)
            else:
                if source in ("oracle", "oracle-bounded"):
                    A, C, delta, positive = oracle_coefficients(basis, g_u, tau)
                    if source == "oracle-bounded":
                        # The model outputs bound * tanh(raw / bound) with unconstrained raw,
                        # so every value strictly inside the bound is reachable: emulate its
                        # range by clipping, not by applying tanh to the targets.
                        cb = config.coefficient_bound * (1 - 1e-9)
                        db = config.delta_bound * (1 - 1e-9)
                        A, C = jnp.clip(A, -cb, cb), jnp.clip(C, -cb, cb)
                        delta = jnp.clip(delta, -db, db)
                else:
                    A = C = jnp.zeros((config.history_length,) * 2)
                    delta = jnp.asarray(0.0)
                direction = apply_inverse_metric(g_u, basis, A, C, delta, tau, config.eps_rel)
            condition = jnp.linalg.cond(dense_inverse_metric(basis, A, C, delta, tau,
                                                             config.eps_rel))
            u_new = jnp.clip(u - s["shrink"] * direction, -1.0, 1.0)
            candidate = jnp.clip(denormalize_coordinates(u_new, lo, hi), lo, hi)
            cv = f(candidate)
            improved = jnp.isfinite(cv) & (cv < s["value"])
            cg = jax.lax.cond(improved, grad_f, jnp.zeros_like, candidate)
            accepted = improved & jnp.all(jnp.isfinite(cg))
            pushed = push_history(s["hist"], normalize_coordinates(candidate, lo, hi),
                                  gradient_to_normalized(cg, lo, hi), cv)
            hist = jax.tree_util.tree_map(lambda a, b: jnp.where(accepted, a, b), pushed,
                                          record_rejection(s["hist"]))
            step_u = normalize_coordinates(candidate, lo, hi) - u
            predicted = jnp.sum(g_u * step_u)
            ratio = jnp.where(jnp.isfinite(cv) & (predicted != 0),
                              (cv - s["value"]) / jnp.where(predicted != 0, predicted, 1.0), 10.0)
            value = jnp.where(accepted, cv, s["value"])
            i = s["iteration"]
            return dict(
                x=jnp.where(accepted, candidate, s["x"]), value=value,
                grad=jnp.where(accepted, cg, s["grad"]), hist=hist,
                shrink=jnp.where(accepted, 1.0, 0.5 * s["shrink"]),
                used=s["used"] + 1 + improved.astype(jnp.int32),
                accepted=s["accepted"] + accepted.astype(jnp.int32),
                rejected=s["rejected"] + (~accepted).astype(jnp.int32), iteration=i + 1,
                rl=jnp.where(accepted, 0.0, jnp.linalg.norm(step_u)),
                rr=jnp.where(accepted, 0.0, jnp.clip(ratio, -10.0, 10.0)), rh=~accepted,
                trajectory=s["trajectory"].at[i + 1].set(value),
                nonpositive=s["nonpositive"] + (~positive).astype(jnp.int32),
                max_condition=jnp.maximum(s["max_condition"], condition),
                max_abs_a=jnp.maximum(s["max_abs_a"], jnp.max(jnp.abs(A))),
                max_abs_c=jnp.maximum(s["max_abs_c"], jnp.max(jnp.abs(C))),
                max_abs_delta=jnp.maximum(s["max_abs_delta"], jnp.abs(delta)),
            )

        return jax.lax.while_loop(keep_going, step, state)

    return run


def lbfgsb(problem_theta: Any, task: Any, budget: int) -> float:
    from scipy.optimize import minimize as scipy_minimize

    value_and_grad = _VG
    calls, best = 0, np.inf

    class Exhausted(Exception):
        pass

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal calls, best
        if calls >= budget // 2:
            raise Exhausted
        calls += 1
        v, g = value_and_grad(x, problem_theta)
        best = min(best, float(v))
        return float(v), np.asarray(g, dtype=float)

    try:
        scipy_minimize(fun, np.asarray(task.x0), jac=True, method="L-BFGS-B",
                       bounds=list(zip(np.asarray(task.lo), np.asarray(task.hi), strict=True)),
                       options={"maxfun": budget // 2, "maxiter": budget, "ftol": 0.0,
                                "gtol": 0.0})
    except Exhausted:
        pass
    return best


_VG = jax.jit(jax.value_and_grad(quadratic_objective))


def parity_check() -> dict[str, Any]:
    """Harness loop with model coefficients == package compiled Gram loop."""
    task = make_task("rotated", 100.0, 10, 5999)
    params = init_gram_model(jax.random.PRNGKey(3), CONFIG)
    ours = make_loop("model", 60, CONFIG)(task.theta, task.x0, task.lo, task.hi, params)
    problem = Problem(lambda x: quadratic_objective(x, task.theta), (task.lo, task.hi), 10,
                      x0=np.asarray(task.x0), dtype=jnp.float64)
    package = minimize(problem, max_evaluations=60, params=params, config=CONFIG,
                       model_kind="gram", candidate_method="preconditioned",
                       preconditioned_engine="compiled")
    n = int(ours["iteration"]) + 1
    np.testing.assert_allclose(np.asarray(ours["trajectory"][:n]), package.trajectory,
                               rtol=1e-10)
    return {"iterations": n - 1, "matched": True}


def tf_h8_cap() -> dict[str, Any]:
    """Anisotropy the frozen v0.6 tf-h8 geometry can express (diag + U Uᵀ)."""
    from neuralopt.training import load_checkpoint

    _, config = load_checkpoint(ROOT / "artifacts" / "trajectory-v0.6.0-tf-h8-seed0.npz")
    bound = config.geometry_bound
    lowrank_max = config.lowrank_scale**2  # ‖U‖_F < lowrank_scale ⇒ λmax(UUᵀ) < scale²
    measured = []
    for seed in range(3):
        params, cfg = load_checkpoint(
            ROOT / "artifacts" / f"trajectory-v0.6.0-tf-h8-seed{seed}.npz")
        for dim in DIMS:
            task = make_task("rotated", 1000.0, dim, 6100 + dim)
            problem = Problem(lambda x, t=task: quadratic_objective(x, t.theta),
                              (task.lo, task.hi), dim, x0=np.asarray(task.x0),
                              dtype=jnp.float64)
            r = minimize(problem, max_evaluations=BUDGET, params=params, config=cfg,
                         candidate_method="preconditioned", preconditioned_engine="compiled",
                         record_diagnostics=True)
            for rec in r.diagnostics["records"]:
                upper = rec["diag_max"] + rec["lowrank_frobenius"] ** 2
                measured.append(upper / rec["diag_min"])
    return {
        "axis_aligned_anisotropy_cap": bound**2,
        "off_axis_anisotropy_cap": (1 / bound + lowrank_max) / (1 / bound),
        "measured_condition_upper_bound_max": float(np.max(measured)),
        "measured_condition_upper_bound_median": float(np.median(measured)),
        "note": "diag in [1/b, b] with b = geometry_bound; lowrank_scale caps λmax(UUᵀ) at "
        "lowrank_scale²; cond(P) <= (diag_max + ‖U‖_F²) / diag_min along tf-h8 runs "
        "(3 seeds x rotated kappa=1000, d = 7/10/16).",
    }


def main() -> None:
    assert jax.config.jax_enable_x64
    parity = parity_check()
    rows = []
    loops = {name: make_loop(name, BUDGET, CONFIG)
             for name in ("oracle", "oracle-bounded", "tau-only")}
    for condition in CONDITIONS:
        for dim in DIMS:
            for r in range(PER_DIM):
                task = make_task("rotated", condition, dim, 6000 + 100 * int(np.log10(condition))
                                 + 10 * dim + r)
                f0 = float(quadratic_objective(task.x0, task.theta))
                row = {"condition": condition, "dimension": dim, "replicate": r}
                for name, loop in loops.items():
                    out = loop(task.theta, task.x0, task.lo, task.hi, {})
                    row[name] = float(np.log10(max(float(out["value"]), 0.0) / f0 + FLOOR))
                    row[f"{name}_units"] = int(out["used"])
                    row[f"{name}_accepted"] = int(out["accepted"])
                    row[f"{name}_iterations"] = int(out["iteration"])
                    row[f"{name}_max_condition"] = float(out["max_condition"])
                    row[f"{name}_nonpositive_steps"] = int(out["nonpositive"])
                    row[f"{name}_max_abs_A"] = float(out["max_abs_a"])
                    row[f"{name}_max_abs_C"] = float(out["max_abs_c"])
                    row[f"{name}_max_abs_delta"] = float(out["max_abs_delta"])
                row["lbfgsb"] = float(np.log10(max(lbfgsb(task.theta, task, BUDGET), 0.0) / f0
                                               + FLOOR))
                rows.append(row)
    summary = {}
    for condition in CONDITIONS:
        sel = [r for r in rows if r["condition"] == condition]
        med = {k: float(np.median([r[k] for r in sel]))
               for k in ("oracle", "oracle-bounded", "tau-only", "lbfgsb")}
        summary[f"kappa_{int(condition)}"] = {
            "median_log10_regret": med,
            "oracle_minus_lbfgsb_median": med["oracle"] - med["lbfgsb"],
            "oracle_reached_1e-6": f"{sum(r['oracle'] <= -6 for r in sel)}/{len(sel)}",
            "lbfgsb_reached_1e-6": f"{sum(r['lbfgsb'] <= -6 for r in sel)}/{len(sel)}",
            "oracle_median_accept_rate": float(np.median(
                [r["oracle_accepted"] / max(r["oracle_iterations"], 1) for r in sel])),
            "oracle_max_condition_median": float(np.median(
                [r["oracle_max_condition"] for r in sel])),
            "oracle_max_abs_A_median": float(np.median([r["oracle_max_abs_A"] for r in sel])),
            "oracle_max_abs_C_median": float(np.median([r["oracle_max_abs_C"] for r in sel])),
            "oracle_max_abs_delta_median": float(np.median(
                [r["oracle_max_abs_delta"] for r in sel])),
            "oracle_max_abs_A_max": float(np.max([r["oracle_max_abs_A"] for r in sel])),
            "oracle_max_abs_C_max": float(np.max([r["oracle_max_abs_C"] for r in sel])),
            "oracle_max_abs_delta_max": float(np.max([r["oracle_max_abs_delta"] for r in sel])),
            "oracle_exceeds_model_bounds_tasks": sum(
                r["oracle_max_abs_A"] >= CONFIG.coefficient_bound
                or r["oracle_max_abs_C"] >= CONFIG.coefficient_bound
                or r["oracle_max_abs_delta"] >= CONFIG.delta_bound for r in sel),
            "gate_pass": med["oracle"] <= med["lbfgsb"] + 1.0,
        }
    output = {
        "schema": "neuralopt-gram-gates/v1",
        "criterion": "oracle median log10 regret <= L-BFGS-B median + 1.0 for kappa 100 and "
        "1000 (pre-stated)",
        "environment": {"jax": jax.__version__, "python": platform.python_version(),
                        "platform": platform.platform(), "float": "float64"},
        "config": CONFIG.__dict__,
        "harness_parity_with_package_loop": parity,
        "gate2_pass": all(v["gate_pass"] for v in summary.values()),
        "summary": summary,
        "tf_h8_anisotropy": tf_h8_cap(),
        "rows": rows,
    }
    RESULTS.write_text(json.dumps(output, indent=1))
    print(json.dumps({k: output[k] for k in ("harness_parity_with_package_loop", "gate2_pass",
                                             "summary", "tf_h8_anisotropy")}, indent=1))


if __name__ == "__main__":
    main()
