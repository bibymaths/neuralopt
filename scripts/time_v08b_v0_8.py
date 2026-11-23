"""v0.8-B timings frozen in its pre-registration (harness only; validation data, no training).

Per tested family (rastrigin, ackley, sinusoid):
  * the policy (untrained parameters; the cost does not depend on the weights): warm
    forward time per step (single and batched), warm rollout time per instance at 100
    units, and the warm time of one ES generation (1,024 episodes at 100 units);
  * each baseline at its AUC-tuned B = 100 configuration from ``results/v0.8-b-b0.json``:
    the median warm cumulative-time curve C_a(t), t = 1..100 units, over 50 validation
    instances run one at a time. Python-level arms are time-stamped per evaluation;
    compiled arms (owned L-BFGS, BB) are timed whole and spread linearly over their units.

Usage: run --family NAME --out RUN_DIR; combine --runs DIR... --output FILE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                      "intra_op_parallelism_threads=1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import run_checkpoint  # noqa: E402
import v08b_baselines as bl  # noqa: E402

from neuralopt.families import FAMILIES, SYNTHETIC, family_objective, make_split  # noqa: E402
from neuralopt.start_predictor import context_stats  # noqa: E402
from neuralopt.step_policy import (  # noqa: E402
    PolicyConfig,
    empty_window,
    init_policy,
    parameter_count,
    policy_outputs,
    policy_rollout,
    push_window,
)

B0 = ROOT / "results" / "v0.8-b-b0.json"
N_TIMING, BUDGET, EPISODES = 50, 100, 1024
RUN_FILES = (ROOT / "scripts" / "time_v08b_v0_8.py", ROOT / "scripts" / "v08b_baselines.py",
             ROOT / "src" / "neuralopt" / "step_policy.py", ROOT / "uv.lock")


def _setup(name: str) -> tuple[Any, Any, Any, Any, PolicyConfig, Any]:
    contexts, _ = make_split(name, "validation", 200)
    if name in SYNTHETIC:
        lo, hi = (np.asarray(b, float) for b in SYNTHETIC[name].bounds("validation"))
        x0 = np.asarray(contexts["x0"])
        config, stats = PolicyConfig(), None
    else:
        spec = FAMILIES[name]
        lo, hi = np.asarray(spec.lo, float), np.asarray(spec.hi, float)
        x0 = np.tile(spec.default, (200, 1))
        train, _ = make_split(name, "train", 2000)
        config, stats = PolicyConfig(context=True), context_stats(train)
    return contexts, lo, hi, x0, config, stats


def _median_curve(curves: list[np.ndarray]) -> list[float]:
    return np.median(np.stack(curves), axis=0).tolist()


def _linear(total: float, used: int) -> np.ndarray:
    t = np.arange(1, BUDGET + 1)
    return total * np.minimum(t, used) / max(used, 1)


def measure(name: str, progress: Any) -> dict[str, Any]:
    contexts, lo, hi, x0, config, stats = _setup(name)
    objective = family_objective(name)
    one = [jax.tree_util.tree_map(lambda a, i=i: a[i], contexts) for i in range(N_TIMING)]
    params = init_policy(jax.random.PRNGKey(0), config)
    out: dict[str, Any] = {"family": name, "policy_parameters": parameter_count(params)}

    # policy: forward per step (single, batched), rollout per instance, one ES generation
    progress("policy", 0, 4)
    d = lo.size
    w = empty_window(d, config)
    for k in range(config.history):
        w = push_window(w, jnp.asarray(np.sin(np.arange(d) + k)), jnp.ones(d), float(k))
    fwd = jax.jit(lambda p, ww, c: policy_outputs(p, ww, 0.5, config, c, stats))
    jax.block_until_ready(fwd(params, w, one[0]))
    singles = []
    for c in one:
        t0 = time.perf_counter()
        jax.block_until_ready(fwd(params, w, c))
        singles.append(time.perf_counter() - t0)
    batched_w = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (EPISODES, *a.shape)), w)
    batched_c = jax.tree_util.tree_map(lambda a: a[np.arange(EPISODES) % 200], contexts)
    fwd_b = jax.jit(jax.vmap(lambda ww, c: policy_outputs(params, ww, 0.5, config, c, stats)))
    jax.block_until_ready(fwd_b(batched_w, batched_c))
    t0 = time.perf_counter()
    jax.block_until_ready(fwd_b(batched_w, batched_c))
    batched_forward = (time.perf_counter() - t0) / EPISODES
    progress("policy", 1, 4)
    roll = jax.jit(lambda c, x, k: policy_rollout(params, objective, c, x, lo, hi, BUDGET, k,
                                                  config, stats)["values"])
    jax.block_until_ready(roll(one[0], jnp.asarray(x0[0]), jax.random.PRNGKey(0)))
    rollouts = []
    for i, c in enumerate(one):
        t0 = time.perf_counter()
        jax.block_until_ready(roll(c, jnp.asarray(x0[i]), jax.random.PRNGKey(i)))
        rollouts.append(time.perf_counter() - t0)
    progress("policy", 2, 4)
    idx = np.arange(EPISODES) % 200
    gen = jax.jit(jax.vmap(lambda c, x, k: policy_rollout(params, objective, c, x, lo, hi, BUDGET,
                                                          k, config, stats)["values"]))
    keys = jax.random.split(jax.random.PRNGKey(1), EPISODES)
    args = (batched_c, jnp.asarray(x0[idx]), keys)
    jax.block_until_ready(gen(*args))
    t0 = time.perf_counter()
    jax.block_until_ready(gen(*args))
    generation = time.perf_counter() - t0
    out["policy"] = {
        "forward_seconds_single_median": float(np.median(singles)),
        "forward_seconds_batched_per_episode": batched_forward,
        "rollout_seconds_per_instance_median": float(np.median(rollouts)),
        "es_generation_seconds_1024_episodes_100_units": generation,
        "cumulative_seconds": _linear(float(np.median(rollouts)), BUDGET).tolist(),
    }

    # baselines at their AUC-tuned B = 100 configuration
    progress("baselines", 3, 4)
    fam = next(f for f in json.loads(B0.read_text())["families"] if f["family"] == name)
    tuned = {arm: v["by_auc"] for arm, v in fam["tuned"]["100"].items()}
    fb = jax.jit(jax.vmap(objective, in_axes=(0, None)))
    f1 = jax.jit(objective)
    vg = jax.jit(jax.value_and_grad(objective))
    curves: dict[str, list[np.ndarray]] = {arm: [] for arm in tuned}

    def stamped(fn: Any, batched: bool) -> tuple[Any, list[float]]:
        stamps: list[float] = []

        def wrapped(x: Any) -> Any:
            value = fn(x)
            now = time.perf_counter()
            stamps.extend([now] * (len(np.atleast_2d(x)) if batched else 1))
            return value
        return wrapped, stamps

    def stamp_curve(stamps: list[float], t0: float, units_per_eval: int) -> np.ndarray:
        cum = np.full(BUDGET, np.nan)
        for j, s in enumerate(stamps):
            u = (j + 1) * units_per_eval
            if u <= BUDGET:
                cum[u - 1] = s - t0
        last = 0.0
        for t in range(BUDGET):  # carry forward between evaluations
            if np.isnan(cum[t]):
                cum[t] = last
            last = cum[t]
        return cum

    for i, c in enumerate(one):
        f_batch = lambda xs, c=c: np.asarray(fb(jnp.asarray(np.atleast_2d(xs)), c))  # noqa: E731
        f_single = lambda x, c=c: float(f1(jnp.asarray(x), c))  # noqa: E731

        def vgrad(x: Any, c: Any = c) -> Any:
            v, g = vg(jnp.asarray(x), c)
            return float(v), np.asarray(g)

        for arm, cfg in tuned.items():
            if i == 0:  # warm-up (compilation) on the first instance, untimed
                _run(arm, cfg, f_batch, f_single, vgrad, objective, c, x0[i], lo, hi, fam)
            if arm in ("quasi-random", "cmaes"):
                fn, stamps = stamped(f_batch, True)
                t0 = time.perf_counter()
                _run(arm, cfg, fn, f_single, vgrad, objective, c, x0[i], lo, hi, fam)
                curves[arm].append(stamp_curve(stamps, t0, 1))
            elif arm == "tpe":
                fn, stamps = stamped(f_single, False)
                t0 = time.perf_counter()
                _run(arm, cfg, f_batch, fn, vgrad, objective, c, x0[i], lo, hi, fam)
                curves[arm].append(stamp_curve(stamps, t0, 1))
            elif arm == "scipy-ms":
                fn, stamps = stamped(vgrad, False)
                t0 = time.perf_counter()
                _run(arm, cfg, f_batch, f_single, fn, objective, c, x0[i], lo, hi, fam)
                curves[arm].append(stamp_curve(stamps, t0, 2))
            else:  # owned-ms, bb: compiled; spread linearly over the units used
                t0 = time.perf_counter()
                used = _run(arm, cfg, f_batch, f_single, vgrad, objective, c, x0[i], lo, hi, fam)
                curves[arm].append(_linear(time.perf_counter() - t0, used))
    out["baselines"] = {arm: {"config": tuned[arm], "cumulative_seconds": _median_curve(v),
                              "seconds_at_100": float(np.median([c[-1] for c in v]))}
                        for arm, v in curves.items()}
    return out


def _run(arm: str, cfg: str, f_batch: Any, f_single: Any, vgrad: Any, objective: Any, c: Any,
         x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, fam: dict[str, Any]) -> int:
    """One B = 100 run of a baseline on one instance; returns units used."""
    kv = dict(p.split("=") for p in cfg.split(",")) if "=" in cfg else {}
    ctx1 = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[None], c)
    if arm == "quasi-random":
        bl.quasi_random(f_batch, x0, lo, hi, BUDGET, seed=0)
        return BUDGET
    if arm == "cmaes":
        pop = None if kv["popsize"] == "default" else 2 * (4 + int(3 * np.log(len(x0))))
        bl.cmaes(f_batch, x0, lo, hi, BUDGET, float(kv["sigma0"]), pop, seed=0)
        return BUDGET
    if arm == "tpe":
        bl.tpe(f_single, x0, lo, hi, BUDGET, int(kv["n_startup"]),
               kv["multivariate"] == "True", seed=0)
        return BUDGET
    if arm == "bb":
        tr = bl.bb(objective, ctx1, x0[None], lo, hi, BUDGET, float(kv["fallback"]))
        return int(tr[0]["objective_calls"] + tr[0]["gradient_calls"])
    k = int(kv["K"])
    per = BUDGET // k
    starts = [x0, *[s for s in _arm_starts(lo, hi, len(x0))[: k - 1]]]
    used = 0
    for s in starts:
        if arm == "scipy-ms":
            tr = bl.scipy_lbfgsb(vgrad, s, lo, hi, per)
        else:
            tr = bl.owned_lbfgs(objective, ctx1, np.asarray(s)[None], lo, hi, max(per, 2))[0]
        used += tr["objective_calls"] + tr["gradient_calls"]
    return used


def _arm_starts(lo: np.ndarray, hi: np.ndarray, d: int) -> np.ndarray:
    from scipy.stats import qmc

    return lo + (hi - lo) * qmc.Sobol(d, scramble=True, seed=3).random_base2(3)[:7]


def run(name: str, run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    code = run_checkpoint.fingerprint(RUN_FILES)
    with (run_dir / "command.txt").open("a") as handle:
        handle.write(run_checkpoint.command_record(
            [sys.executable, *sys.argv],
            {"code_fingerprint": code, "cap": "0.5 CPU-hours (process CPU clock; one pinned "
                                              "core); 1.5 CPU-hours for the three families"}))
    run_checkpoint.run_units([name], run_dir, measure, fingerprint=code)
    (run_dir / "DONE").write_text("units/\n")
    return 0


def combine(dirs: list[Path], output: Path) -> int:
    if output.exists():
        raise SystemExit(f"{output} exists; results are never overwritten")
    rows = [json.loads(p.read_text())["row"]
            for d in dirs for p in sorted((d / "units").glob("*.json"))]
    run_checkpoint.write_atomic(output, json.dumps({"schema": "neuralopt-v08b-timing/v1",
                                                    "families": rows}, indent=1) + "\n")
    print(f"wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=("rastrigin", "ackley", "sinusoid"), required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("combine")
    c.add_argument("--runs", type=Path, nargs="+", required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.family, args.out) if args.command == "run" else combine(args.runs, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
