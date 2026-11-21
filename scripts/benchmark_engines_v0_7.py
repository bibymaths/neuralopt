"""Track A6: wall-clock benchmark of inference engines and references (float64, CPU).

Each arm runs in a fresh process so "cold" includes compilation. Columns:
cold = first solve in a fresh process (tracing + compilation + run);
warm = median single solve afterwards; batched = tasks per second over the task set
(``vmap`` where the arm supports it, otherwise a sequential loop).
All timings use ``block_until_ready`` / host conversion of the result.

Tasks: rotated kappa=100 quadratics, d=16, interior optima (box [-5, 5]); budget 100
objective+gradient units. Optimistix has no box bounds for multivariate minimisers; with
interior optima the box is inactive at the solution, but iterates may leave it.

Usage::

    uv run python scripts/benchmark_engines_v0_7.py            # all arms
    uv run python scripts/benchmark_engines_v0_7.py --arm NAME # one arm (JSON to stdout)
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ["JAX_ENABLE_X64"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from neuralopt import Problem, minimize  # noqa: E402
from neuralopt.compiled import preconditioned_solve, preconditioned_solve_batch  # noqa: E402
from neuralopt.task_families import make_task, quadratic_objective  # noqa: E402
from neuralopt.training import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "engine-benchmark-v0.7.0.json"
CHECKPOINT = ROOT / "artifacts" / "trajectory-v0.6.0-tf-h8-seed0.npz"
DIMENSION, CONDITION, BUDGET, TASKS, WARM_REPEATS = 16, 100.0, 100, 32, 10
FIXED_STEP = 0.03  # the v0.6 validation-tuned fixed step
ARMS = (
    "tf-h8-python",
    "tf-h8-compiled",
    "fixed-compiled",
    "scipy-lbfgsb",
    "optimistix-bfgs",
    "optimistix-lbfgs",
)


def tasks() -> list[Any]:
    return [make_task("rotated", CONDITION, DIMENSION, 5000 + i) for i in range(TASKS)]


def stacked(ts: list[Any]) -> tuple[Any, Any, Any, Any]:
    thetas = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[t.theta for t in ts])
    return thetas, jnp.stack([t.x0 for t in ts]), jnp.stack([t.lo for t in ts]), jnp.stack(
        [t.hi for t in ts]
    )


def log_regret(value: float, f0: float) -> float:
    return float(np.log10(max(value, 0.0) / f0 + 1e-16))


def run_arm(arm: str) -> dict[str, Any]:
    assert jax.config.jax_enable_x64
    ts = tasks()
    closures = [(lambda x, th=t.theta: quadratic_objective(x, th)) for t in ts]
    f0 = [float(c(t.x0)) for c, t in zip(closures, ts, strict=True)]
    params, config = load_checkpoint(CHECKPOINT)
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), params)
    problems = [
        Problem(c, (t.lo, t.hi), DIMENSION, x0=np.asarray(t.x0), dtype=jnp.float64)
        for c, t in zip(closures, ts, strict=True)
    ]
    counts: list[tuple[int, int]] = []
    # Per task: (units after each iteration, incumbent value after each iteration).
    traces: dict[int, tuple[list[int], list[float]]] = {}
    iterations: dict[int, int] = {}

    def loop_trace(i: int, values: list[float]) -> None:
        """NeuralOpt loop: start costs 2; a candidate costs 1, plus 1 if it improved."""
        units, used = [], 2
        for prev, value in zip(values[:-1], values[1:], strict=True):
            used += 1 + int(value < prev)
            units.append(used)
        traces[i] = (units, list(values[1:]))
        iterations[i] = len(values) - 1

    if arm in ("tf-h8-python", "tf-h8-compiled", "fixed-compiled"):
        learned = arm.startswith("tf-h8")
        engine = "python" if arm == "tf-h8-python" else "compiled"

        def solve(i: int) -> float:
            if engine == "python":  # only accepts Problem closures (recompiles per call)
                r = minimize(
                    problems[i], max_evaluations=BUDGET, params=params, config=config,
                    candidate_method="preconditioned", preconditioned_engine=engine,
                    hamiltonian_metric="learned" if learned else "fixed",
                    fixed_metric_diag=np.ones(DIMENSION), preconditioned_step=FIXED_STEP,
                )
                d = r.diagnostics or {}
                counts.append((d["objective_evaluations"], d["gradient_evaluations"]))
                loop_trace(i, list(r.trajectory))
                return r.fun
            # Parametric objective: one compilation for every task of this family/dimension.
            out = preconditioned_solve(
                quadratic_objective, ts[i].x0, (ts[i].lo, ts[i].hi), BUDGET,
                theta=ts[i].theta, params=params, config=config,
                geometry="learned" if learned else "fixed",
                fixed_metric_diag=np.ones(DIMENSION), preconditioned_step=FIXED_STEP,
            )
            counts.append((int(out["objective_evaluations"]), int(out["gradient_evaluations"])))
            loop_trace(i, [float(v) for v in out["trajectory"][: int(out["iteration"]) + 1]])
            return float(out["value"])

        batch: Any = None
        if engine == "compiled":
            thetas, x0, lo, hi = stacked(ts)

            def batch() -> Any:
                out = preconditioned_solve_batch(
                    quadratic_objective, thetas, x0, (lo, hi), BUDGET, params=params,
                    config=config, geometry="learned" if learned else "fixed",
                    fixed_metric_diag=np.ones(DIMENSION), preconditioned_step=FIXED_STEP,
                )
                return out["fun"]

    elif arm == "scipy-lbfgsb":
        from scipy.optimize import minimize as scipy_minimize

        value_and_grad = jax.jit(jax.value_and_grad(quadratic_objective))
        batch = None

        def solve(i: int) -> float:
            theta = ts[i].theta
            calls, best = 0, np.inf
            bests: list[float] = []

            class Exhausted(Exception):
                pass

            def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
                nonlocal calls, best
                if calls >= BUDGET // 2:
                    raise Exhausted
                calls += 1
                v, g = value_and_grad(x, theta)
                v = float(v)
                best = min(best, v)
                bests.append(best)
                return v, np.asarray(g, dtype=float)

            t = ts[i]
            try:
                scipy_minimize(
                    fun, np.asarray(t.x0), jac=True, method="L-BFGS-B",
                    bounds=list(zip(np.asarray(t.lo), np.asarray(t.hi), strict=True)),
                    options={"maxfun": BUDGET // 2, "maxiter": BUDGET, "ftol": 0.0, "gtol": 0.0},
                )
            except Exhausted:
                pass
            counts.append((calls, calls))
            traces[i] = ([2 * (k + 1) for k in range(calls)], bests)
            iterations[i] = calls
            return best

    else:
        import optimistix as optx

        solver_cls = optx.BFGS if arm == "optimistix-bfgs" else optx.LBFGS
        solver = solver_cls(rtol=1e-14, atol=1e-14)

        def fn(y: Any, theta: Any) -> Any:
            return quadratic_objective(y, theta)

        def one(theta: Any, x0: Any) -> Any:
            # One fn evaluation (value + linearization) per step, line-search trials
            # included: max_steps = budget / 2 units of objective+gradient.
            sol = optx.minimise(fn, solver, x0, args=theta, max_steps=BUDGET // 2, throw=False)
            return fn(sol.value, theta), sol.stats["num_steps"]

        single = jax.jit(one)
        thetas, x0s, _, _ = stacked(ts)
        batched = jax.jit(jax.vmap(one))

        def solve(i: int) -> float:
            value, steps = single(ts[i].theta, ts[i].x0)
            counts.append((int(steps), int(steps)))
            iterations[i] = int(steps)
            return float(value)

        def optimistix_traces() -> None:
            """Incumbent after k steps for k = 1..50, via max_steps = k (measurement only;
            the extra fn(sol.value) used for reporting is not part of the solve)."""
            per_k = []
            for k in range(1, BUDGET // 2 + 1):
                def run_k(theta: Any, x0: Any, k: int = k) -> Any:
                    sol = optx.minimise(fn, solver, x0, args=theta, max_steps=k, throw=False)
                    return fn(sol.value, theta)
                per_k.append(np.asarray(jax.jit(jax.vmap(run_k))(thetas, x0s)))
            values_k = np.minimum.accumulate(np.stack(per_k), axis=0)  # best so far
            for i in range(TASKS):
                traces[i] = ([2 * k for k in range(1, BUDGET // 2 + 1)], values_k[:, i].tolist())

        def count_evaluations() -> bool:
            """Untimed check: fn is called exactly once per step (line-search trials too)."""
            calls = {"n": 0}

            def counting_fn(y: Any, theta: Any) -> Any:
                jax.debug.callback(lambda: calls.__setitem__("n", calls["n"] + 1))
                return quadratic_objective(y, theta)

            ok = True
            for i in range(2):
                calls["n"] = 0
                sol = optx.minimise(counting_fn, solver, ts[i].x0, args=ts[i].theta,
                                    max_steps=BUDGET // 2, throw=False)
                jax.block_until_ready(sol.value)
                ok &= calls["n"] == int(sol.stats["num_steps"])
            return ok

        def batch() -> Any:
            values, _ = batched(thetas, x0s)
            return jax.block_until_ready(values)

    start = time.perf_counter()
    first = solve(0)
    cold = time.perf_counter() - start
    warm = []
    for repeat in range(WARM_REPEATS):  # new task each time, same family and dimension
        start = time.perf_counter()
        solve(1 + repeat)
        warm.append(time.perf_counter() - start)
    values = [solve(i) for i in range(TASKS)]  # every task, also for quality and counts
    if batch is not None:
        batch()  # compile
        start = time.perf_counter()
        batched_values = np.asarray(batch())
        batched_seconds = time.perf_counter() - start
        batch_mode = "vmap"
    else:
        start = time.perf_counter()
        batched_values = np.asarray([solve(i) for i in range(TASKS)])
        batched_seconds = time.perf_counter() - start
        batch_mode = "sequential"
    regrets = [log_regret(v, f) for v, f in zip(values, f0, strict=True)]
    extra: dict[str, Any] = {}
    if arm.startswith("optimistix"):
        optimistix_traces()
        extra["optimistix_fn_calls_equal_steps_verified"] = count_evaluations()
        extra["optimistix_budget_enforcement"] = (
            "max_steps=50; each step makes exactly one fn evaluation with its linearization "
            "(objective + gradient = 2 units), whether the line search accepts or rejects the "
            "trial point; verified with a call counter"
        )
    time_per_iteration = float(np.median([
        w / iterations[1 + r] for r, w in enumerate(warm) if iterations.get(1 + r)
    ]))
    targets = {}
    for target in (1e-3, 1e-6):
        units_hit, iters_hit = [], []
        for i in range(TASKS):
            units, vals = traces[i]
            hit = next((k for k, v in enumerate(vals) if v / f0[i] <= target), None)
            if hit is not None:
                units_hit.append(units[hit])
                iters_hit.append(hit + 1)
        reached = len(units_hit)
        targets[f"{target:g}"] = {
            "reached": f"{reached}/{TASKS}",
            "median_units_to_target": float(np.median(units_hit)) if reached else None,
            "median_iterations_to_target": float(np.median(iters_hit)) if reached else None,
            # Estimated as iterations-to-target x warm time per iteration.
            "estimated_ms_to_target": float(np.median(iters_hit)) * time_per_iteration * 1e3
            if reached else None,
        }
    task_counts = counts[-TASKS:] if batch_mode == "vmap" else counts[-2 * TASKS : -TASKS]
    return {
        "arm": arm,
        "cold_seconds": cold,
        "warm_seconds_median": float(np.median(warm)),
        "warm_ms_per_budget_unit": float(np.median(warm)) / BUDGET * 1e3,
        "batched_tasks_per_second": TASKS / batched_seconds,
        "batch_mode": batch_mode,
        "batched_matches_single": bool(np.allclose(batched_values, values, rtol=1e-8)),
        "median_log10_regret": float(np.median(regrets)),
        "mean_objective_evaluations": float(np.mean([c[0] for c in task_counts])),
        "mean_gradient_evaluations": float(np.mean([c[1] for c in task_counts])),
        "max_units": int(max(c[0] + c[1] for c in task_counts)),
        "first_value_finite": bool(np.isfinite(first)),
        "mean_units_used": float(np.mean([c[0] + c[1] for c in task_counts])),
        "mean_iterations": float(np.mean([iterations[i] for i in range(TASKS)])),
        "warm_ms_per_iteration": time_per_iteration * 1e3,
        "iteration_definition": "one candidate (1 unit, +1 if improved)"
        if arm in ("tf-h8-python", "tf-h8-compiled", "fixed-compiled")
        else "one objective+gradient call (2 units)",
        "targets": targets,
        **extra,
    }


def profile_compiled(repeats: int = 2000) -> dict[str, Any]:
    """Split a warm compiled tf-h8 solve into fixed overhead, model forward and f/grad.

    fixed overhead = median budget-2 solve (initial f, gradient, history push, dispatch,
    host transfer; zero iterations). Per-iteration cost = (full - fixed) / iterations.
    Model forward and objective/gradient costs are measured inside one compiled
    ``fori_loop`` of ``repeats`` evaluations (no dispatch), with a carried dependency
    so XLA cannot hoist them.
    """
    from neuralopt.features import gradient_to_normalized, normalize_coordinates
    from neuralopt.model import apply_model
    from neuralopt.trajectory import empty_history, history_token_features, push_history

    ts = tasks()
    params, config = load_checkpoint(CHECKPOINT)
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), params)

    def solve(i: int, budget: int, geometry: str = "learned") -> dict[str, Any]:
        return preconditioned_solve(
            quadratic_objective, ts[i].x0, (ts[i].lo, ts[i].hi), budget, theta=ts[i].theta,
            params=params, config=config, geometry=geometry,
            fixed_metric_diag=np.ones(DIMENSION), preconditioned_step=FIXED_STEP,
        )

    def timed(fn: Any) -> float:
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start

    out: dict[str, Any] = {}
    for geometry in ("learned", "fixed"):
        solve(0, BUDGET, geometry), solve(0, 2, geometry)  # compile both budgets
        full, iters = [], []
        for i in range(1, 11):
            full.append(timed(lambda i=i, g=geometry: solve(i, BUDGET, g)))
            iters.append(int(solve(i, BUDGET, geometry)["iteration"]))
        fixed = float(np.median([timed(lambda i=i, g=geometry: solve(i, 2, g))
                                 for i in range(1, 11)]))
        per_iteration = float(np.median([(f - fixed) / n for f, n in zip(full, iters,
                                                                          strict=True)]))
        out[geometry] = {
            "median_solve_ms": float(np.median(full)) * 1e3,
            "median_iterations": float(np.median(iters)),
            "fixed_per_call_overhead_ms": fixed * 1e3,
            "per_iteration_us": per_iteration * 1e6,
        }
    task = ts[0]
    lo, hi, theta = task.lo, task.hi, task.theta
    x = task.x0 + 0.1
    grad = jax.grad(quadratic_objective)(x, theta)
    hist = empty_history(config.history_length, DIMENSION, jnp.float64)
    for k in range(config.history_length + 1):
        xk = x + 0.01 * k
        hist = push_history(hist, normalize_coordinates(xk, lo, hi), gradient_to_normalized(
            jax.grad(quadratic_objective)(xk, theta), lo, hi), quadratic_objective(xk, theta))

    @jax.jit
    def model_loop(x: Any) -> Any:
        def body(_: Any, carry: Any) -> Any:
            tokens = history_token_features(carry, grad, lo, hi, hist)
            o = apply_model(params, tokens, 0.5, config)
            return carry + 1e-30 * (o["diag"] + o["step"])
        return jax.lax.fori_loop(0, repeats, body, x)

    @jax.jit
    def objective_loop(x: Any) -> Any:
        def body(_: Any, carry: Any) -> Any:
            return carry + 1e-30 * quadratic_objective(carry, theta)
        return jax.lax.fori_loop(0, repeats, body, x)

    @jax.jit
    def gradient_loop(x: Any) -> Any:
        def body(_: Any, carry: Any) -> Any:
            return carry + 1e-30 * jax.grad(quadratic_objective)(carry, theta)
        return jax.lax.fori_loop(0, repeats, body, x)

    for name, fn in (("model_forward", model_loop), ("objective", objective_loop),
                     ("gradient", gradient_loop)):
        jax.block_until_ready(fn(x))
        seconds = np.median([timed(lambda fn=fn: jax.block_until_ready(fn(x))) for _ in range(5)])
        out[f"{name}_us_compiled"] = float(seconds / repeats * 1e6)
    learned = out["learned"]
    accept_rate = 36.4375 / 62.75  # mean gradient/objective calls of the tf-h8 runs (A6)
    model_us = out["model_forward_us_compiled"]
    fg_us = out["objective_us_compiled"] + accept_rate * out["gradient_us_compiled"]
    out["learned_per_iteration_split_us"] = {
        "model_forward": model_us,
        "objective_and_gradient": fg_us,
        "loop_machinery_and_rest": learned["per_iteration_us"] - model_us - fg_us,
        "model_floor_ms_per_solve": model_us * learned["median_iterations"] / 1e3,
    }
    return out


TRAIN_VARIANTS = (("isotropic", 1.0), ("diagonal", 10.0), ("diagonal", 100.0),
                  ("diagonal", 1000.0), ("rotated", 10.0), ("rotated", 100.0), ("rastrigin", 0.0))


def training_benchmark(updates: int = 60) -> dict[str, Any]:
    """v0.6 trainer (one Problem closure per task) vs the compiled batched trainer.

    Same 21 tasks as the v0.6 training split (seeds, scales), float64, same model and
    unroll. Reports wall time including compilation, and per-update and per-task-gradient
    rates after compilation.
    """
    from neuralopt.fast_training import train_batched
    from neuralopt.model import ModelConfig
    from neuralopt.task_families import family_objective, stack_tasks
    from neuralopt.training import HistoryUnrollConfig

    config = ModelConfig(width=16, heads=4, rank=2, layers=1, scale_aware=True,
                         geometry_bound=10.0, lowrank_scale=0.5, feature_set="history-v1",
                         history_length=8)
    specs = []
    for i, (family, condition) in enumerate(TRAIN_VARIANTS):
        for j, dim in enumerate((2, 4, 8)):
            index = i + j
            scale = (0.2, 1.0, 5.0)[index % 3]
            coordinate = (0.5, 1.0, 2.0)[(2 * index + 1) % 3]
            specs.append(make_task(family, condition, dim, 1000 + 10 * i + j, scale, coordinate))
    problems = [
        Problem(lambda x, s=s: family_objective(s.family)(x, s.theta), (s.lo, s.hi),
                s.dimension, dtype=jnp.float64, metadata={"known_optimum": 0.0})
        for s in specs
    ]
    unroll = HistoryUnrollConfig()
    out: dict[str, Any] = {"tasks": len(specs), "updates": updates, "unroll_steps": unroll.steps}
    # Steady state of the v0.6 trainer, reproducing train()'s loop exactly: one jitted
    # value_and_grad per Problem closure (all compiled first, timed separately), then
    # per update: loss+gradient call, finiteness checks, eager optax update.
    import optax

    from neuralopt.model import init_model
    from neuralopt.training import history_regret_loss

    params = init_model(jax.random.PRNGKey(0), config)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    state = optimizer.init(params)
    start = time.perf_counter()
    compiled = []
    for problem in problems:
        fn = jax.jit(jax.value_and_grad(
            lambda p, k, problem=problem: history_regret_loss(p, problem, k, config, unroll)))
        jax.block_until_ready(fn(params, jax.random.PRNGKey(0)))
        compiled.append(fn)
    compile_seconds = time.perf_counter() - start
    key = jax.random.PRNGKey(1)
    start = time.perf_counter()
    for step in range(updates):
        key, sub = jax.random.split(key)
        loss, grads = compiled[step % len(compiled)](params, sub)
        if not bool(jnp.isfinite(loss)) or not all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(grads)
        ):
            raise FloatingPointError("nonfinite v0.6 training step")
        updates_, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates_)
        float(loss)
    steady = time.perf_counter() - start
    out["v06_train"] = {
        "compilations": len(problems),
        "compile_seconds": compile_seconds,
        "seconds_for_updates_after_compile": steady,
        "updates_per_second": updates / steady,
        "task_gradients_per_second": updates / steady,
    }
    groups = stack_tasks(specs)
    for batch in (1, 8):
        start = time.perf_counter()
        train_batched(groups, steps=len(groups), batch_size=batch, config=config, unroll=unroll)
        compile_pass = time.perf_counter() - start
        report = train_batched(groups, steps=updates, batch_size=batch, config=config,
                               unroll=unroll)
        per_task = sum(min(batch, groups[s % len(groups)].size) for s in range(updates))
        out[f"batched_train_batch{batch}"] = {
            "compilations": len(groups),
            "group_sizes": {g.name: g.size for g in groups},
            "first_pass_seconds_incl_compile": compile_pass,
            "seconds_for_updates_after_compile": report.seconds,
            "updates_per_second": updates / report.seconds,
            "task_gradients_per_second": per_task / report.seconds,
            "skipped_updates": report.skipped_updates,
        }
    return out


def main() -> None:
    rows = []
    for arm in ARMS:
        out = subprocess.run(
            [sys.executable, __file__, "--arm", arm], capture_output=True, text=True, check=True
        )
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]))
    RESULTS.write_text(
        json.dumps(
            {
                "schema": "neuralopt-engine-benchmark/v1",
                "environment": {
                    "jax": jax.__version__,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                    "xla_flags": os.environ.get("XLA_FLAGS"),
                    "float": "float64",
                },
                "protocol": {
                    "tasks": f"{TASKS} rotated kappa={CONDITION:g} quadratics, d={DIMENSION}, "
                    "interior optima, seeds 5000+",
                    "budget_units": BUDGET,
                    "warm_repeats": WARM_REPEATS,
                    "checkpoint": CHECKPOINT.name,
                    "optimistix_note": "unbounded solvers; iterates may leave the box",
                    "warm_note": "median over 10 solves of new tasks of the same family; the "
                    "python engine takes Problem closures and recompiles objective/gradient on "
                    "every call (A1)",
                },
                "rows": rows,
                "profile_compiled": json.loads(
                    subprocess.run([sys.executable, __file__, "--profile"], capture_output=True,
                                   text=True, check=True).stdout.strip().splitlines()[-1]
                ),
                "training": json.loads(
                    subprocess.run([sys.executable, __file__, "--training"], capture_output=True,
                                   text=True, check=True).stdout.strip().splitlines()[-1]
                ),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.profile:
        print(json.dumps(profile_compiled()))
    elif args.training:
        print(json.dumps(training_benchmark()))
    elif args.arm:
        print(json.dumps(run_arm(args.arm)))
    else:
        main()
