"""v0.7 mask diagnostic for the oracle (harness only, no training).

Frozen loop: halving, K = 10, float64, budget 100; gate tasks (rotated kappa 100/1000,
d = 7/10/16, 4 per d, seeds 6000+).

PRE-STATED decision rule (recorded before running, 2026-11-22): relaxed masks close a
meaningful part of the gap iff the better relaxed variant reduces the kappa = 1000
median gap (oracle K10 - SciPy maxcor 10) by >= 1.0 decade relative to the current masks.

Mask variants (pairs whose two endpoints are both stored, i.e. "filled"):
  current : finite, max|s| > 1e-7 (u units), ‖y‖ > 1e-6 · max(‖g_k‖, ‖g_k+1‖)  [package]
  float64 : finite, max|s| > 1e-12,          ‖y‖ > 1e-10 · max(‖g_k‖, ‖g_k+1‖)
  minimal : finite and sᵀy > 0 only
Per run: masked fraction of filled pairs by reason under the *current* thresholds (step,
gradient change, nonfinite; first failing reason in that order) and the effective number
of valid pairs per iteration for the variant actually used.
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

from neuralopt.features import gradient_to_normalized, normalize_coordinates  # noqa: E402
from neuralopt.gram import (  # noqa: E402
    GramBasis,
    GramConfig,
    apply_inverse_metric,
    gram_basis,
    tau_scale,  # noqa: E402
)
from neuralopt.loop_rules import accept_step  # noqa: E402
from neuralopt.task_families import quadratic_objective  # noqa: E402
from neuralopt.trajectory import empty_history, push_history, record_rejection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gram_gates_v0_7 import oracle_coefficients  # noqa: E402
from run_loop_rule_ablation_v0_7 import lbfgsb, task_list  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "mask-diagnostic-v0.7.0.json"
BUDGET, PAIRS, FLOOR, AMENDED_FLOOR, DRAWS = 100, 10, 1e-16, -12.0, 10_000
THRESHOLDS = {"current": (1e-7, 1e-6), "float64": (1e-12, 1e-10)}
_TINY = 1e-30


def _norm(x: Any) -> Any:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=-1) + _TINY)


def variant_basis(h: Any, mode: str) -> tuple[GramBasis, dict[str, Any]]:
    s = h.u[1:] - h.u[:-1]
    y = h.g[1:] - h.g[:-1]
    filled = h.valid[1:] & h.valid[:-1]
    finite = (jnp.all(jnp.isfinite(s), axis=-1) & jnp.all(jnp.isfinite(y), axis=-1)
              & jnp.isfinite(h.f[1:]) & jnp.isfinite(h.f[:-1]))
    scale = jnp.maximum(_norm(h.g[1:]), _norm(h.g[:-1]))
    step_tol, grad_rtol = THRESHOLDS["current"]
    small_step = jnp.max(jnp.abs(s), axis=-1, initial=0.0) <= step_tol
    small_change = _norm(y) <= grad_rtol * scale
    reasons = {  # under the current thresholds, first failing reason
        "filled": jnp.sum(filled),
        "nonfinite": jnp.sum(filled & ~finite),
        "step": jnp.sum(filled & finite & small_step),
        "gradient_change": jnp.sum(filled & finite & ~small_step & small_change),
    }
    if mode == "minimal":
        mask = filled & finite & (jnp.sum(s * y, axis=-1) > 0)
    else:
        st, gr = THRESHOLDS[mode]
        mask = (filled & finite & (jnp.max(jnp.abs(s), axis=-1, initial=0.0) > st)
                & (_norm(y) > gr * scale))
    keep = mask[:, None]
    s, y = jnp.where(keep, s, 0.0), jnp.where(keep, y, 0.0)
    s_norm, y_norm = _norm(s), _norm(y)
    basis = GramBasis(
        jnp.where(keep, s / s_norm[:, None], 0.0).T,
        jnp.where(keep, y / y_norm[:, None], 0.0).T,
        jnp.where(mask, s_norm, 0.0),
        jnp.where(mask, y_norm, 0.0),
        mask,
        jnp.where(mask, jnp.sum(s * y, axis=-1), 0.0),
    )
    return basis, reasons


def make_loop(mode: str) -> Any:
    config = GramConfig(history_length=PAIRS)

    @jax.jit
    def run(theta: Any, x0: Any, lo: Any, hi: Any) -> dict[str, Any]:
        f = lambda x: quadratic_objective(x, theta)  # noqa: E731
        grad_f = jax.grad(f)
        value, grad = f(x0), grad_f(x0)
        hist = push_history(empty_history(PAIRS, x0.shape[0], x0.dtype),
                            normalize_coordinates(x0, lo, hi),
                            gradient_to_normalized(grad, lo, hi), value)
        i32 = jnp.int32
        zeros = jnp.zeros((BUDGET,), i32)
        state = dict(x=x0, value=value, grad=grad, hist=hist, shrink=jnp.asarray(1.0),
                     used=jnp.asarray(2, i32), iterations=jnp.asarray(0, i32),
                     filled=zeros, nonfinite=zeros, step=zeros, gradient_change=zeros,
                     valid=zeros)

        def keep_going(s: dict[str, Any]) -> Any:
            return (s["used"] + 2 <= BUDGET) & (
                jnp.linalg.norm(gradient_to_normalized(s["grad"], lo, hi)) != 0)

        def body(s: dict[str, Any]) -> dict[str, Any]:
            g_u = gradient_to_normalized(s["grad"], lo, hi)
            basis, reasons = variant_basis(s["hist"], mode)
            tau = tau_scale(basis, g_u, config.fallback_length)
            A, C, delta, _ = oracle_coefficients(basis, g_u, tau)
            base = apply_inverse_metric(g_u, basis, A, C, delta, tau, config.eps_rel)
            out = accept_step("halving", f, grad_f, s["x"], s["value"], s["grad"], base,
                              jnp.asarray(1.0), s["shrink"], BUDGET - s["used"], lo, hi)
            pushed = push_history(s["hist"], normalize_coordinates(out.x, lo, hi),
                                  gradient_to_normalized(out.grad, lo, hi), out.value)
            hist = jax.tree_util.tree_map(lambda a, b: jnp.where(out.accepted, a, b), pushed,
                                          record_rejection(s["hist"]))
            i = s["iterations"]
            return dict(
                x=out.x, value=out.value, grad=out.grad, hist=hist,
                shrink=out.next_shrink.astype(x0.dtype),
                used=s["used"] + out.objective_evaluations + out.gradient_evaluations,
                iterations=i + 1,
                valid=s["valid"].at[i].set(jnp.sum(basis.mask)),
                **{k: s[k].at[i].set(reasons[k]) for k in
                   ("filled", "nonfinite", "step", "gradient_change")},
            )

        return jax.lax.while_loop(keep_going, body, state)

    return run


def check_current_matches_package() -> bool:
    from neuralopt.task_families import make_task

    task = make_task("rotated", 1000.0, 7, 5997)
    rng = np.random.default_rng(0)
    h = empty_history(PAIRS, 7, jnp.float64)
    for k in range(PAIRS + 1):
        x = jnp.asarray(rng.normal(size=7)) * (0.5 if k % 3 else 1e-9)
        h = push_history(h, normalize_coordinates(x, task.lo, task.hi),
                         gradient_to_normalized(jax.grad(quadratic_objective)(x, task.theta),
                                                task.lo, task.hi),
                         quadratic_objective(x, task.theta))
    ours, _ = variant_basis(h, "current")
    package = gram_basis(h)
    return all(bool(jnp.array_equal(a, b)) for a, b in zip(ours, package, strict=True))


def paired(differences: np.ndarray, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, (DRAWS, differences.size), replace=True).mean(axis=1)
    return {"mean": float(differences.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "median": float(np.median(differences))}


def main() -> None:
    assert jax.config.jax_enable_x64
    assert check_current_matches_package(), "current variant must equal the package masks"
    loops = {mode: make_loop(mode) for mode in ("current", "float64", "minimal")}
    rows = []
    for condition, dim, rep, task in task_list():
        f0 = float(quadratic_objective(task.x0, task.theta))
        best, _ = lbfgsb(task, 10)
        row: dict[str, Any] = {
            "condition": condition, "dimension": dim, "replicate": rep,
            "lbfgsb_m10": float(np.log10(max(best, 0.0) / f0 + FLOOR)),
        }
        for mode, loop in loops.items():
            out = loop(task.theta, task.x0, task.lo, task.hi)
            n = int(out["iterations"])
            filled = int(np.sum(out["filled"][:n]))
            row[mode] = {
                "log10_regret": float(np.log10(max(float(out["value"]), 0.0) / f0 + FLOOR)),
                "iterations": n,
                "masked_fraction_of_filled": {
                    k: (int(np.sum(out[k][:n])) / filled if filled else 0.0)
                    for k in ("step", "gradient_change", "nonfinite")
                },
                "mean_valid_pairs": float(np.mean(out["valid"][:n])) if n else 0.0,
                "final_valid_pairs": int(out["valid"][n - 1]) if n else 0,
                "valid_pairs_by_iteration": [int(v) for v in out["valid"][:n]],
            }
        rows.append(row)
    summary: dict[str, Any] = {}
    for condition in (100.0, 1000.0):
        sel = [r for r in rows if r["condition"] == condition]
        ref = np.array([r["lbfgsb_m10"] for r in sel])
        block: dict[str, Any] = {"lbfgsb_m10_median": float(np.median(ref)),
                                 "lbfgsb_m10_reached_1e-6": f"{int(np.sum(ref <= -6))}/{len(sel)}"}
        for mode in loops:
            values = np.array([r[mode]["log10_regret"] for r in sel])
            block[mode] = {
                "median_log10_regret": float(np.median(values)),
                "reached_1e-6": f"{int(np.sum(values <= -6))}/{len(sel)}",
                "median_gap_to_lbfgsb_m10": float(np.median(values) - np.median(ref)),
                "paired_difference_raw": paired(values - ref),
                "paired_difference_floored_-12": paired(
                    np.maximum(values, AMENDED_FLOOR) - np.maximum(ref, AMENDED_FLOOR)),
                "masked_fraction_of_filled_mean": {
                    k: float(np.mean([r[mode]["masked_fraction_of_filled"][k] for r in sel]))
                    for k in ("step", "gradient_change", "nonfinite")
                },
                "mean_valid_pairs": float(np.mean([r[mode]["mean_valid_pairs"] for r in sel])),
            }
        summary[f"kappa_{int(condition)}"] = block
    gap_current = summary["kappa_1000"]["current"]["median_gap_to_lbfgsb_m10"]
    best_relaxed = min(summary["kappa_1000"][m]["median_gap_to_lbfgsb_m10"]
                       for m in ("float64", "minimal"))
    decision = {
        "rule": "relaxed masks close a meaningful part iff the better relaxed variant "
        "reduces the kappa=1000 median gap by >= 1.0 decade (pre-stated)",
        "gap_current": gap_current,
        "gap_best_relaxed": best_relaxed,
        "reduction": gap_current - best_relaxed,
        "meaningful": gap_current - best_relaxed >= 1.0,
    }
    output = {"schema": "neuralopt-mask-diagnostic/v1",
              "environment": {"jax": jax.__version__, "python": platform.python_version(),
                              "platform": platform.platform(), "float": "float64"},
              "frozen_loop": "halving, K=10, compiled-engine rule via accept_step, budget 100",
              "summary": summary, "decision": decision, "rows": rows}
    RESULTS.write_text(json.dumps(output, indent=1))
    print(json.dumps({"summary": summary, "decision": decision}, indent=1))


if __name__ == "__main__":
    main()
