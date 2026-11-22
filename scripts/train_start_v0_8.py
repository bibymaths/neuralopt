"""Main-track training of the start predictor g_φ (pre-registration §5) and the
validation-only choices. Harness only. No test or OOD split is opened here.

Subcommands:
  labels    --family F --out DIR                 training targets for train and validation
                                                 (owned multistart, 16 starts x 1,000 units)
  train     --family F --loss {R,O} --seed S --labels DIR --out DIR
  validate  --family F --runs DIR... --labels DIR --out DIR
            A4 loss choice, tau_fit, tau_c, diversity, and the frozen per-unit and
            forward-pass timings for amendment 1 (validation only)
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
import f0_instances_v0_8 as fi  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import run_checkpoint  # noqa: E402
import run_f0_v0_8 as f0  # noqa: E402

from neuralopt.families import FAMILIES, family_objective, make_split  # noqa: E402
from neuralopt.lbfgs import lbfgs_solve_batch  # noqa: E402
from neuralopt.start_predictor import (  # noqa: E402
    StartConfig,
    candidate_diversity,
    context_stats,
    encode,
    init_start_model,
    load_start_checkpoint,
    objective_loss,
    observation_tokens,
    predict_starts,
    regression_loss,
    save_start_checkpoint,
    start_candidates,
    start_confidence,
)

MATCHED_B = {"sinusoid": 200, "cascade-clean": 400, "cascade-realistic": 800}
STEPS, BATCH, LR, LR_END, CLIP, EVAL_EVERY = 10_000, 64, 1e-3, 1e-5, 1.0, 500
CONF_STEPS, CONF_BATCH, CONF_LR = 2_000, 256, 1e-3
SILENT_MISS_TARGET = 0.005
TIMING_INSTANCES = 50
INSTANCES = ROOT / "results" / "f0-instances-v0.8.0.json"
RUN_FILES = (ROOT / "scripts" / "train_start_v0_8.py", ROOT / "scripts" / "run_f0_v0_8.py",
             ROOT / "scripts" / "f0_instances_v0_8.py", ROOT / "scripts" / "run_checkpoint.py",
             ROOT / "src" / "neuralopt" / "start_predictor.py",
             ROOT / "src" / "neuralopt" / "families.py", ROOT / "src" / "neuralopt" / "lbfgs.py",
             ROOT / "uv.lock")


def _record(run_dir: Path, cap: str) -> str:
    code = run_checkpoint.fingerprint(RUN_FILES)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "command.txt").open("a") as handle:
        handle.write(run_checkpoint.command_record([sys.executable, *sys.argv],
                                                   {"code_fingerprint": code, "cap": cap}))
    return code


def _finish(run_dir: Path, summary: dict[str, Any]) -> None:
    run_checkpoint.write_atomic(run_dir / "result.json", json.dumps(summary, indent=1) + "\n")
    (run_dir / "DONE").write_text("result.json\n")


# ---- labels ---------------------------------------------------------------------------


def labels(name: str, out: Path) -> int:
    _record(out, "0.5 CPU-hours (process CPU clock; one pinned core)")
    spec = FAMILIES[name]
    starts = f0._sobol(spec, f0.LABEL_STARTS, f0.SOBOL_SEEDS["arms"] + 10)

    def compute(split: str, progress: Any) -> dict[str, Any]:
        n = f0.N_TRAIN if split == "train" else f0.N_VALIDATION
        contexts, _ = make_split(name, split, n)
        best_val, best_x = np.full(n, np.inf), np.zeros((n, spec.dimension))
        for k in range(f0.LABEL_STARTS):
            progress(f"{split} label start", k, f0.LABEL_STARTS)
            for i, r in enumerate(f0._owned_traces(name, contexts, np.tile(starts[k], (n, 1)),
                                                   f0.LABEL_UNITS)):
                if r["trace"][-1][1] < best_val[i]:
                    best_val[i], best_x[i] = r["trace"][-1][1], r["x"]
        np.savez(out / f"labels-{split}.npz", x=best_x, f=best_val)
        return {"split": split, "instances": n, "f_median": float(np.median(best_val))}

    run_checkpoint.run_units(["train", "validation"], out, compute,
                             fingerprint=run_checkpoint.fingerprint(RUN_FILES))
    _finish(out, {"family": name})
    return 0


# ---- training -------------------------------------------------------------------------


def _stack(contexts: dict[str, Any], idx: Any) -> dict[str, Any]:
    return jax.tree_util.tree_map(lambda a: a[idx], contexts)


def train(name: str, loss_kind: str, seed: int, label_dir: Path, out: Path) -> int:
    _record(out, "3 CPU-hours for this process (process CPU clock; one pinned core); "
                 "training total 26 CPU-hours (amendment 3)")
    spec = FAMILIES[name]
    lo, hi = jnp.asarray(spec.lo), jnp.asarray(spec.hi)
    config = StartConfig(dimension=spec.dimension)
    objective = family_objective(name)
    train_ctx, _ = make_split(name, "train", f0.N_TRAIN)
    val_ctx, _ = make_split(name, "validation", f0.N_VALIDATION)
    tl, vl = np.load(label_dir / "labels-train.npz"), np.load(label_dir / "labels-validation.npz")
    train_x, train_f = jnp.asarray(tl["x"]), jnp.asarray(tl["f"])
    val_x = jnp.asarray(vl["x"])
    stats = context_stats(train_ctx)
    f_scale = float(np.median(np.asarray(train_f)))
    code = run_checkpoint.fingerprint(RUN_FILES)

    def instance_loss(params: Any, c: Any, target: Any) -> Any:
        pooled = encode(params, observation_tokens(c, stats), c["mask"], config)
        cands = start_candidates(params, pooled, lo, hi, config)
        if loss_kind == "R":
            return regression_loss(cands, target, lo, hi)
        return objective_loss(cands, objective, c, f_scale)

    def batch_loss(params: Any, ctx: Any, targets: Any) -> Any:
        return jnp.mean(jax.vmap(instance_loss, in_axes=(None, 0, 0))(params, ctx, targets))

    schedule = optax.cosine_decay_schedule(LR, STEPS, alpha=LR_END / LR)
    optimizer = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(schedule))

    @jax.jit
    def step(params: Any, state: Any, ctx: Any, targets: Any) -> tuple[Any, Any, Any]:
        value, grads = jax.value_and_grad(batch_loss)(params, ctx, targets)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value

    val_loss = jax.jit(batch_loss)

    def stage_candidates(_: str, progress: Any) -> dict[str, Any]:
        params = init_start_model(jax.random.PRNGKey(seed), config)
        state = optimizer.init(params)
        rng = np.random.default_rng(seed)
        best, best_val, history = params, np.inf, []
        t0, c0 = time.perf_counter(), time.process_time()
        for s in range(1, STEPS + 1):
            idx = rng.integers(0, f0.N_TRAIN, BATCH)
            params, state, value = step(params, state, _stack(train_ctx, idx), train_x[idx])
            if s % 100 == 0:
                progress("candidates", s, STEPS)
            if s % EVAL_EVERY == 0:
                v = float(val_loss(params, val_ctx, val_x))
                history.append([s, float(value), v])
                if v < best_val:
                    best_val, best = v, params
        save_start_checkpoint(out / "candidates.npz", best, config, stats,
                              {"family": name, "loss": loss_kind, "seed": seed})
        return {"best_validation_loss": best_val, "history": history,
                "wall_seconds": time.perf_counter() - t0,
                "cpu_seconds": time.process_time() - c0, "f_scale": f_scale}

    def stage_confidence(_: str, progress: Any) -> dict[str, Any]:
        t0, c0 = time.perf_counter(), time.process_time()
        params, _, _ = load_start_checkpoint(out / "candidates.npz")
        b = MATCHED_B[name]

        @jax.jit
        def pooled_and_cands(c: Any) -> tuple[Any, Any]:
            pooled = encode(params, observation_tokens(c, stats), c["mask"], config)
            return pooled, start_candidates(params, pooled, lo, hi, config)

        pooled, cands = jax.vmap(pooled_and_cands)(train_ctx)  # (n, 2w), (n, K, d)
        labels_ok = np.zeros((f0.N_TRAIN, config.candidates), bool)
        n = f0.N_TRAIN
        bounds = (jnp.tile(lo, (n, 1)), jnp.tile(hi, (n, 1)))
        for k in range(config.candidates):
            progress("confidence labels", k, config.candidates)
            res = lbfgs_solve_batch(objective, train_ctx, cands[:, k], bounds, b, strategy="map")
            labels_ok[:, k] = [f0._success(float(v), float(fs))
                               for v, fs in zip(res["fun"], np.asarray(train_f), strict=True)]
        conf_keys = [k for k in params if k.startswith("conf_")]
        conf = {k: params[k] for k in conf_keys}
        frozen = {k: v for k, v in params.items() if k not in conf_keys}
        conf_opt = optax.adam(CONF_LR)
        cstate = conf_opt.init(conf)
        flat_p = jnp.repeat(pooled, config.candidates, axis=0)
        flat_c = cands.reshape(-1, config.dimension)
        flat_y = jnp.asarray(labels_ok.reshape(-1), float)

        def bce(cparams: Any, p: Any, c: Any, y: Any) -> Any:
            prob = jax.vmap(lambda pp, cc: start_confidence({**frozen, **cparams}, pp, cc[None],
                                                            lo, hi)[0])(p, c)
            prob = jnp.clip(prob, 1e-12, 1 - 1e-12)
            return -jnp.mean(y * jnp.log(prob) + (1 - y) * jnp.log(1 - prob))

        @jax.jit
        def cstep(cparams: Any, state: Any, p: Any, c: Any, y: Any) -> tuple[Any, Any, Any]:
            value, grads = jax.value_and_grad(bce)(cparams, p, c, y)
            updates, state = conf_opt.update(grads, state, cparams)
            return optax.apply_updates(cparams, updates), state, value

        rng = np.random.default_rng(seed + 1000)
        for s in range(1, CONF_STEPS + 1):
            idx = rng.integers(0, flat_y.shape[0], CONF_BATCH)
            conf, cstate, value = cstep(conf, cstate, flat_p[idx], flat_c[idx], flat_y[idx])
            if s % 200 == 0:
                progress("confidence head", s, CONF_STEPS)
        final = {**frozen, **conf}
        save_start_checkpoint(out / "start.npz", final, config, stats,
                              {"family": name, "loss": loss_kind, "seed": seed})
        return {"label_success_rate": float(labels_ok.mean()), "final_bce": float(value),
                "wall_seconds": time.perf_counter() - t0,
                "cpu_seconds": time.process_time() - c0}

    stages = {"candidates": stage_candidates, "confidence": stage_confidence}
    rows = run_checkpoint.run_units(list(stages), out, lambda u, p: stages[u](u, p),
                                    fingerprint=code, weights={"candidates": 4.0})
    _finish(out, {"family": name, "loss": loss_kind, "seed": seed,
                  "candidates": {k: v for k, v in rows[0].items() if k != "history"},
                  "confidence": rows[1],
                  "cpu_seconds": rows[0]["cpu_seconds"] + rows[1]["cpu_seconds"]})
    return 0


# ---- validation-only choices ----------------------------------------------------------


def validate(name: str, runs: list[Path], label_dir: Path, out: Path) -> int:
    _record(out, "1 CPU-hour (process CPU clock; one pinned core)")
    spec = FAMILIES[name]
    lo, hi = jnp.asarray(spec.lo), jnp.asarray(spec.hi)
    b = MATCHED_B[name]
    n = f0.N_VALIDATION
    val_ctx, _ = make_split(name, "validation", n)
    objective = family_objective(name)
    row = next(r for r in json.loads(INSTANCES.read_text())["families"] if r["family"] == name)
    f_star = np.asarray(row["per_instance"]["f_star"])
    tol = f0.EPS_ABS + f0.EPS_REL * np.abs(f_star)
    tau_fit = float(np.quantile(f_star, 0.99))
    bounds = (jnp.tile(lo, (n, 1)), jnp.tile(hi, (n, 1)))
    per_run: dict[str, Any] = {}
    for run in runs:
        meta = json.loads((run / "result.json").read_text())
        params, config, stats = load_start_checkpoint(run / "start.npz")
        pred = jax.jit(jax.vmap(lambda c, p=params, s=stats, cf=config:
                                predict_starts(p, c, s, lo, hi, cf)))(val_ctx)
        top = pred["order"][:, 0]
        x0 = np.asarray(pred["candidates"])[np.arange(n), np.asarray(top)]
        conf_top = np.asarray(pred["confidence"])[np.arange(n), np.asarray(top)]
        traces = f0._owned_traces(name, val_ctx, x0, f0.LONG)
        units = [fi._units_to_success(t["trace"], f_star[i]) for i, t in enumerate(traces)]
        at_b = np.array([u is not None and u <= b for u in units])
        refined_b = lbfgs_solve_batch(objective, val_ctx, jnp.asarray(x0), bounds, b,
                                      strategy="map")
        fit_b = np.asarray(refined_b["fun"])
        diversity = float(np.median([float(candidate_diversity(c, lo, hi))
                                     for c in np.asarray(pred["candidates"])]))
        per_run[f"{meta['loss']}/seed={meta['seed']}"] = {
            "loss": meta["loss"], "seed": meta["seed"], "success_at_B": float(at_b.mean()),
            "units_to_success": units, "confidence_top": conf_top.tolist(),
            "fit_at_B": fit_b.tolist(), "success_at_B_per_instance": at_b.tolist(),
            "diversity_median": diversity, "collapsed": diversity < 0.05}
    by_loss = {loss: float(np.mean([r["success_at_B"] for r in per_run.values()
                                    if r["loss"] == loss])) for loss in ("R", "O")}
    chosen = "O" if by_loss["O"] > by_loss["R"] else "R"
    runs_c = [r for r in per_run.values() if r["loss"] == chosen]
    thresholds = sorted({0.0, *[c for r in runs_c for c in r["confidence_top"]], 1.01})
    tau_c, rate_c, fallback_c = None, None, None
    for tau in thresholds:
        silent, fallback = [], []
        for r in runs_c:
            conf = np.asarray(r["confidence_top"])
            trig = (conf < tau) | (np.asarray(r["fit_at_B"]) > tau_fit)
            ok = np.asarray(r["fit_at_B"]) - f_star <= tol
            silent.append(np.mean(~trig & ~ok))
            fallback.append(np.mean(trig))
        rate = float(np.mean(silent))
        if rate <= SILENT_MISS_TARGET:
            tau_c, rate_c, fallback_c = float(tau), rate, float(np.mean(fallback))
            break
    timing = _timings(name, runs, chosen, val_ctx, lo, hi)
    summary = {"family": name, "matched_budget": b, "success_at_B_by_loss": by_loss,
               "chosen_loss": chosen, "tau_fit": tau_fit, "tau_c": tau_c,
               "validation_silent_miss_at_tau_c": rate_c,
               "validation_fallback_rate_at_tau_c": fallback_c, "timing": timing,
               "runs": per_run}
    _finish(out, summary)
    return 0


def _timings(name: str, runs: list[Path], chosen: str, val_ctx: Any, lo: Any, hi: Any
             ) -> dict[str, Any]:
    """Frozen-before-test warm timings (amendment 1): seconds per unit for owned L-BFGS and
    full-budget SciPy (one instance at a time), and the g_φ forward time."""
    spec = FAMILIES[name]
    objective = family_objective(name)
    ctx = [jax.tree_util.tree_map(lambda a, i=i: a[i], val_ctx) for i in range(TIMING_INSTANCES)]
    one_lo, one_hi = jnp.asarray(spec.lo)[None], jnp.asarray(spec.hi)[None]
    single = [jax.tree_util.tree_map(lambda a, i=i: a[i:i + 1], val_ctx)
              for i in range(TIMING_INSTANCES)]
    x0 = jnp.asarray(spec.default)[None]
    lbfgs_solve_batch(objective, single[0], x0, (one_lo, one_hi), f0.LONG, strategy="map")
    owned = []
    for c in single:
        t0 = time.perf_counter()
        res = lbfgs_solve_batch(objective, c, x0, (one_lo, one_hi), f0.LONG, strategy="map")
        jax.block_until_ready(res["fun"])
        owned.append((time.perf_counter() - t0) / max(int(res["used"][0]), 1))
    vg = jax.jit(jax.value_and_grad(objective))
    fi.scipy_full_trace(vg, spec.default, ctx[0], spec, f0.LONG)
    scipy = []
    for c in ctx:
        t0 = time.perf_counter()
        tr = fi.scipy_full_trace(vg, spec.default, c, spec, f0.LONG)
        scipy.append((time.perf_counter() - t0) / max(2 * len(tr), 1))
    run = next(r for r in runs if json.loads((r / "result.json").read_text())["loss"] == chosen)
    params, config, stats = load_start_checkpoint(run / "start.npz")
    fwd = jax.jit(lambda c: predict_starts(params, c, stats, lo, hi, config)["candidates"])
    jax.block_until_ready(fwd(ctx[0]))
    forward = []
    for c in ctx:
        t0 = time.perf_counter()
        jax.block_until_ready(fwd(c))
        forward.append(time.perf_counter() - t0)
    return {"instances": TIMING_INSTANCES,
            "owned_lbfgs_seconds_per_unit_median": float(np.median(owned)),
            "scipy_full_seconds_per_unit_median": float(np.median(scipy)),
            "forward_seconds_median": float(np.median(forward))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("labels")
    a.add_argument("--family", choices=list(MATCHED_B), required=True)
    a.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--family", choices=list(MATCHED_B), required=True)
    t.add_argument("--loss", choices=("R", "O"), required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--labels", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("validate")
    v.add_argument("--family", choices=list(MATCHED_B), required=True)
    v.add_argument("--runs", type=Path, nargs="+", required=True)
    v.add_argument("--labels", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "labels":
        return labels(args.family, args.out)
    if args.command == "train":
        return train(args.family, args.loss, args.seed, args.labels, args.out)
    return validate(args.family, args.runs, args.labels, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
