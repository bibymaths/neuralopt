"""v0.8 main track: the single test and OOD evaluation (harness).

Frozen protocol: ``docs/research/v0.8-preregistration.md`` with amendments 1–4
(``docs/research/v0.8-preregistration-amendments.md``). Validation-only choices come from
``results/v0.8-validation-choices.json``; the trained checkpoints come from the training run
directory. Nothing here tunes or selects anything.

Subcommands:
  verify-splits             frozen split hashes (test at the frozen size; OOD; validation)
  shard --family F --split S --start a --stop b --out DIR
                            per-instance outcomes of every arm on instances [a, b) of a
                            split, checkpointed in chunks of 25 (resumable)
  timing --family F --out DIR
                            C10: warm and cold forward and refinement times, single and
                            batched, on the first 200 test instances. Runs alone.

Per instance the shard records: the reference optimum (two Sobol seeds) and its argmin; the
final f* after the update rule; units to success (≤ 800, else None) for A0, A1, A1-full,
A3, A2/A2-full/A2o per Sobol start, and the learned arms per training seed (A4, the A4K
ranks, A5); A4's top confidence and refined fit at B (for A4F); prediction errors (C9);
objective and gradient calls.
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
import run_checkpoint  # noqa: E402
import run_f0_v0_8 as f0  # noqa: E402  (sets float64)

from neuralopt.families import FAMILIES, family_objective, make_split, split_digest  # noqa: E402
from neuralopt.lbfgs import lbfgs_solve_batch  # noqa: E402
from neuralopt.start_predictor import load_start_checkpoint, predict_starts  # noqa: E402

TRAIN_RUN = ROOT / "runs" / "train-20261122T2238Z"
CHOICES = ROOT / "results" / "v0.8-validation-choices.json"
MATCHED_B = {"sinusoid": 200, "cascade-clean": 400, "cascade-realistic": 800}
TEST_SIZE = {"sinusoid": 7000, "cascade-clean": 1200, "cascade-realistic": 300}
TEST_SHA = {
    "sinusoid": "e9491fa8ce0c60005f1afd5ed21c954d24448c5ac504a3b58430001163635bb9",
    "cascade-clean": "426fe2eec2c84dea04d446d8a986db28f423a8fd88dd933ed954e26553844db4",
    "cascade-realistic": "46a097548a250093a52a1dba3a16f7f5564f10430e97bd604286bd291b65e2be",
}
OOD_SPLITS = {"sinusoid": ("ood-noise", "ood-params", "ood-sampling"),
              "cascade-clean": ("ood-noise", "ood-params", "ood-sampling", "ood-dose"),
              "cascade-realistic": ("ood-noise", "ood-params", "ood-sampling", "ood-dose")}
OOD_SIZE = 200
SEEDS = (0, 1, 2, 3, 4)
C7_FAMILIES = ("cascade-realistic",)  # §3: the only family F0 flags as non-identified
CHUNK = 25
LONG = 800
RUN_FILES = (ROOT / "scripts" / "evaluate_v0_8.py", ROOT / "scripts" / "run_f0_v0_8.py",
             ROOT / "scripts" / "f0_instances_v0_8.py", ROOT / "scripts" / "run_checkpoint.py",
             ROOT / "src" / "neuralopt" / "families.py", ROOT / "src" / "neuralopt" / "lbfgs.py",
             ROOT / "src" / "neuralopt" / "start_predictor.py", CHOICES, ROOT / "uv.lock")


def rank_budgets(b: int) -> list[int]:
    """Budget of the owned run from each confidence rank 1..8 (A4K reads K = 1, 2, 4, 8 off
    these: rank 1 runs to 800 for A4; ranks 2, 3–4, 5–8 need only B/2, B/4, B/8)."""
    return [LONG, b // 2, b // 4, b // 4, b // 8, b // 8, b // 8, b // 8]


def to_u(x: Any, lo: Any, hi: Any) -> np.ndarray:
    return 2.0 * (np.asarray(x) - lo) / (hi - lo) - 1.0


def _finite(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def _precise_best(runner: Any, starts: list[Any], c: Any) -> tuple[float, np.ndarray]:
    """F0's reference runs (``_ScipyRunner.trace(precise=True)``, 2,000 units each), also
    keeping the point of the lowest value."""
    best_value, best_x = np.inf, np.asarray(starts[0], float)
    for x0 in starts:
        record: dict[str, Any] = {"v": np.inf, "x": None}
        vg = runner.vg

        def tracking(x: Any, c: Any, _vg: Any = vg, _r: dict[str, Any] = record) -> Any:
            v, g = _vg(x, c)
            fv = float(v)
            if np.isfinite(fv) and fv < _r["v"]:
                _r["v"], _r["x"] = fv, np.asarray(x, float).copy()
            return v, g

        runner.vg = tracking
        try:
            tr = runner.trace(x0, c, f0.REFERENCE_UNITS, precise=True)
        finally:
            runner.vg = vg
        if tr and tr[-1][1] < best_value:
            best_value, best_x = tr[-1][1], record["x"]
    return best_value, best_x


class Models:
    """The trained start predictors (validation-chosen loss, plus the other loss on C7's
    family), jitted over a batch of contexts."""

    def __init__(self, name: str) -> None:
        choice = next(f for f in json.loads(CHOICES.read_text())["families"]
                      if f["family"] == name)
        spec = FAMILIES[name]
        lo, hi = jnp.asarray(spec.lo), jnp.asarray(spec.hi)
        self.chosen = choice["chosen_loss"]
        self.losses = [self.chosen] + ([("O" if self.chosen == "R" else "R")]
                                       if name in C7_FAMILIES else [])
        self.predict: dict[tuple[str, int], Any] = {}
        for loss in self.losses:
            for seed in SEEDS:
                params, config, stats = load_start_checkpoint(
                    TRAIN_RUN / name / f"{loss}-seed{seed}" / "start.npz")
                self.predict[(loss, seed)] = jax.jit(jax.vmap(
                    lambda c, p=params, s=stats, cf=config: predict_starts(p, c, s, lo, hi, cf)))


def _nearest_starts(name: str, contexts: Any) -> np.ndarray:
    """A3: F0's nearest-neighbour start, with labels from the training run's label stage
    (the F0 label procedure: same starts, budget and code)."""
    labels = np.load(TRAIN_RUN / name / "labels" / "labels-train.npz")
    train, _ = make_split(name, "train", f0.N_TRAIN)
    feats = f0._nn_features(name, train)
    mean, std = feats.mean(0), feats.std(0) + 1e-12
    z_train = (feats - mean) / std
    z = (f0._nn_features(name, contexts) - mean) / std
    nearest = np.argmin(((z[:, None, :] - z_train[None, :, :]) ** 2).sum(-1), axis=1)
    return labels["x"][nearest]


def evaluate_chunk(name: str, contexts: Any, truths: Any, a3_start: np.ndarray,
                   models: Models, offset: int, progress: Any) -> list[dict[str, Any]]:
    spec = FAMILIES[name]
    lo, hi = np.asarray(spec.lo), np.asarray(spec.hi)
    n = int(np.asarray(truths).shape[0])
    b = MATCHED_B[name]
    ctx = [jax.tree_util.tree_map(lambda a, i=i: a[i], contexts) for i in range(n)]
    scipy = f0._ScipyRunner(name)
    vg = jax.jit(jax.value_and_grad(family_objective(name)))
    finals: list[list[float]] = [[] for _ in range(n)]

    def note(i: int, value: float) -> None:
        if value is not None and np.isfinite(value):
            finals[i].append(float(value))

    # reference optimum (F0 protocol; both seeds; argmin kept for C9)
    progress("reference", 0, n)
    ref = {"a": np.full(n, np.inf), "b": np.full(n, np.inf)}
    x_star = np.zeros((n, spec.dimension))
    for tag, key in (("reference_a", "a"), ("reference_b", "b")):
        starts = f0._sobol(spec, f0.REFERENCE_STARTS, f0.SOBOL_SEEDS[tag])
        for i in range(n):
            v, x = _precise_best(scipy, [*starts, np.asarray(truths[i])], ctx[i])
            ref[key][i] = v
            if key == "a" or v < ref["a"][i]:
                x_star[i] = x if x is not None else x_star[i]

    # generic arms
    progress("generic arms", 0, n)
    a0 = f0._owned_traces(name, contexts, np.tile(spec.default, (n, 1)), LONG)
    a3 = f0._owned_traces(name, contexts, a3_start, LONG)
    a1 = [scipy.trace(spec.default, ctx[i], LONG) for i in range(n)]
    a1_full = [fi.scipy_full_trace(vg, spec.default, ctx[i], spec, LONG) for i in range(n)]
    sobol = f0._sobol(spec, max(f0.MULTISTART_K), f0.SOBOL_SEEDS["arms"])
    a2 = [[scipy.trace(s, ctx[i], LONG) for s in sobol] for i in range(n)]
    a2_full = [[fi.scipy_full_trace(vg, s, ctx[i], spec, LONG) for s in sobol] for i in range(n)]
    a2o = [f0._owned_traces(name, contexts, np.tile(s, (n, 1)), LONG) for s in sobol]

    # learned arms
    progress("learned arms", 0, n)
    budgets = rank_budgets(b)
    learned: dict[str, dict[str, Any]] = {}
    for loss in models.losses:
        per_seed = []
        for seed in SEEDS:
            # one instance at a time (batch 1): the forward pass differs at rounding level
            # between batch sizes, so this keeps results independent of shard layout
            preds = [models.predict[(loss, seed)](jax.tree_util.tree_map(
                lambda a, i=i: a[i:i + 1], contexts)) for i in range(n)]
            cands = np.concatenate([np.asarray(p["candidates"]) for p in preds])
            conf = np.concatenate([np.asarray(p["confidence"]) for p in preds])
            order = np.concatenate([np.asarray(p["order"]) for p in preds])
            ranks = range(8) if loss == models.chosen else range(1)
            runs = [f0._owned_traces(name, contexts, cands[np.arange(n), order[:, r]], budgets[r])
                    for r in ranks]
            a5 = ([fi.scipy_full_trace(vg, cands[i, order[i, 0]], ctx[i], spec, b)
                   for i in range(n)] if loss == models.chosen else None)
            per_seed.append({"cands": cands, "conf": conf, "order": order, "runs": runs,
                             "a5": a5})
        learned[loss] = {"seeds": per_seed}

    # f* = min(reference, every evaluated final), the F0 update rule
    for i in range(n):
        note(i, ref["a"][i])
        note(i, ref["b"][i])
        for t in (a0[i], a3[i], *[a2o[k][i] for k in range(len(sobol))]):
            note(i, t["trace"][-1][1])
        for t in (a1[i], a1_full[i], *a2[i], *a2_full[i]):
            if t:
                note(i, t[-1][1])
        for loss in learned.values():
            for s in loss["seeds"]:
                for r in s["runs"]:
                    note(i, r[i]["trace"][-1][1])
                if s["a5"] is not None and s["a5"][i]:
                    note(i, s["a5"][i][-1][1])
    rows = []
    for i in range(n):
        ref_best = min(ref["a"][i], ref["b"][i])
        f_star = min(finals[i]) if finals[i] else np.inf
        tol = f0.EPS_ABS + f0.EPS_REL * abs(ref_best)

        def u(trace: Any, fs: float = f_star) -> int | None:
            return fi._units_to_success(trace, fs)

        u_star = to_u(x_star[i], lo, hi)
        row: dict[str, Any] = {
            "index": offset + i,
            "reference_a": _finite(ref["a"][i]), "reference_b": _finite(ref["b"][i]),
            "reference_reproduced": bool(abs(ref["a"][i] - ref["b"][i]) <= f0.EPS_ABS
                                         + f0.EPS_REL * abs(min(ref["a"][i], ref["b"][i]))),
            "f_star": _finite(f_star), "f_star_lowered_by_an_arm": bool(f_star < ref_best - tol),
            "x_star": x_star[i].tolist(),
            "A0": {"u": u(a0[i]["trace"]), "obj": a0[i]["objective_evaluations"],
                   "grad": a0[i]["gradient_evaluations"]},
            "A3": {"u": u(a3[i]["trace"]), "obj": a3[i]["objective_evaluations"],
                   "grad": a3[i]["gradient_evaluations"],
                   "error": float(np.linalg.norm(to_u(a3_start[i], lo, hi) - u_star))},
            "A1": {"u": u(a1[i]), "calls": len(a1[i])},
            "A1_full": {"u": u(a1_full[i]), "calls": len(a1_full[i])},
            "A2": [u(t) for t in a2[i]],
            "A2_full": [u(t) for t in a2_full[i]],
            "A2o": [u(a2o[k][i]["trace"]) for k in range(len(sobol))],
        }
        for loss, data in learned.items():
            entry: dict[str, list[Any]] = {"conf_top": [], "conf_all": [], "u": [],
                                           "fit_at_B": [], "error": [], "obj": [], "grad": []}
            if loss == models.chosen:
                entry.update(rank_u=[], A5_u=[], A5_calls=[])
            for s in data["seeds"]:
                top = int(s["order"][i, 0])
                r1 = s["runs"][0][i]
                entry["conf_top"].append(float(s["conf"][i, top]))
                entry["conf_all"].append(s["conf"][i].tolist())
                entry["u"].append(u(r1["trace"]))
                entry["fit_at_B"].append(float(f0._best_within(r1["trace"], b)))
                entry["error"].append(float(np.linalg.norm(to_u(s["cands"][i, top], lo, hi)
                                                           - u_star)))
                entry["obj"].append(r1["objective_evaluations"])
                entry["grad"].append(r1["gradient_evaluations"])
                if loss == models.chosen:
                    entry["rank_u"].append([u(run[i]["trace"]) for run in s["runs"]])
                    entry["A5_u"].append(u(s["a5"][i]))
                    entry["A5_calls"].append(len(s["a5"][i]))
            row[f"learned_{loss}"] = entry
        rows.append(row)
    return rows


def shard(name: str, split: str, start: int, stop: int, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    code = run_checkpoint.fingerprint(RUN_FILES)
    with (out / "command.txt").open("a") as handle:
        handle.write(run_checkpoint.command_record(
            [sys.executable, *sys.argv],
            {"code_fingerprint": code,
             "cap": "3 CPU-hours for this process (process CPU clock; one pinned core); "
                    "evaluation total 24 CPU-hours (amendment 4)"}))
    contexts, truths = make_split(name, split, stop)
    sl = slice(start, stop)
    contexts = jax.tree_util.tree_map(lambda a: a[sl], contexts)
    truths = np.asarray(truths)[sl]
    a3_start = _nearest_starts(name, contexts)
    models = Models(name)
    names = [f"chunk-{k:05d}" for k in range(start, stop, CHUNK)]

    def compute(unit: str, progress: Any) -> dict[str, Any]:
        k0 = int(unit.split("-")[1]) - start
        k1 = min(k0 + CHUNK, stop - start)
        part = jax.tree_util.tree_map(lambda a: a[k0:k1], contexts)
        c0 = time.process_time()
        rows = evaluate_chunk(name, part, truths[k0:k1], a3_start[k0:k1], models, start + k0,
                              progress)
        return {"rows": rows, "cpu_seconds": time.process_time() - c0}

    run_checkpoint.run_units(names, out, compute, fingerprint=code)
    run_checkpoint.write_atomic(out / "process_cpu.json", json.dumps(
        {"process_cpu_seconds": time.process_time()}) + "\n")
    (out / "DONE").write_text("units/\n")
    return 0


def verify_splits(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "results" / "f0-v0.8.0.json").read_text())["splits"]
    report, ok = {}, True
    for name in MATCHED_B:
        got = split_digest(*make_split(name, "test", TEST_SIZE[name]))
        report[f"{name}/test/{TEST_SIZE[name]}"] = got == TEST_SHA[name]
        for split in ("validation", "test", *OOD_SPLITS[name]):
            n = frozen[name][split]["instances"]
            report[f"{name}/{split}/{n}"] = split_digest(*make_split(name, split, n)) == \
                frozen[name][split]["sha256"]
    ok = all(report.values())
    run_checkpoint.write_atomic(out / "result.json", json.dumps({"ok": ok, "checks": report},
                                                                indent=1) + "\n")
    print(json.dumps(report, indent=1))
    if ok:
        (out / "DONE").write_text("result.json\n")
    return 0 if ok else 1


def timing(name: str, out: Path) -> int:
    """C10, measured in one process: forward and owned-L-BFGS refinement (budget B, from
    A4's top start), warm one instance at a time, cold (first call), and batched."""
    out.mkdir(parents=True, exist_ok=True)
    code = run_checkpoint.fingerprint(RUN_FILES)
    with (out / "command.txt").open("a") as handle:
        handle.write(run_checkpoint.command_record(
            [sys.executable, *sys.argv],
            {"code_fingerprint": code, "cap": "0.5 CPU-hours (process CPU clock; one pinned "
                                              "core); runs alone"}))
    spec = FAMILIES[name]
    lo, hi = jnp.asarray(spec.lo), jnp.asarray(spec.hi)
    b = MATCHED_B[name]
    n = 200
    contexts, _ = make_split(name, "test", n)
    objective = family_objective(name)
    chosen = next(f for f in json.loads(CHOICES.read_text())["families"]
                  if f["family"] == name)["chosen_loss"]
    one_lo, one_hi = lo[None], hi[None]
    single = [jax.tree_util.tree_map(lambda a, i=i: a[i:i + 1], contexts) for i in range(n)]
    rows = {}
    c0 = time.process_time()
    for seed in SEEDS:
        params, config, stats = load_start_checkpoint(TRAIN_RUN / name / f"{chosen}-seed{seed}"
                                                      / "start.npz")
        fwd = jax.jit(jax.vmap(lambda c, p=params, s=stats, cf=config:
                               predict_starts(p, c, s, lo, hi, cf)))
        t0 = time.perf_counter()
        first = fwd(single[0])
        jax.block_until_ready(first)
        cold_forward = time.perf_counter() - t0
        x0 = jnp.asarray(np.asarray(first["candidates"])[0, int(first["order"][0, 0])])[None]
        t0 = time.perf_counter()
        jax.block_until_ready(lbfgs_solve_batch(objective, single[0], x0, (one_lo, one_hi), b,
                                                strategy="map")["fun"])
        cold_refine = time.perf_counter() - t0
        forward, refine, used = [], [], []
        for c in single:
            t0 = time.perf_counter()
            pred = fwd(c)
            jax.block_until_ready(pred)
            forward.append(time.perf_counter() - t0)
            x0 = jnp.asarray(np.asarray(pred["candidates"])[0, int(pred["order"][0, 0])])[None]
            t0 = time.perf_counter()
            res = lbfgs_solve_batch(objective, c, x0, (one_lo, one_hi), b, strategy="map")
            jax.block_until_ready(res["fun"])
            refine.append(time.perf_counter() - t0)
            used.append(int(res["used"][0]))
        t0 = time.perf_counter()
        pred = fwd(contexts)
        jax.block_until_ready(pred)
        batched_forward = (time.perf_counter() - t0) / n
        cands = np.asarray(pred["candidates"])[np.arange(n), np.asarray(pred["order"])[:, 0]]
        bounds = (jnp.tile(lo, (n, 1)), jnp.tile(hi, (n, 1)))
        lbfgs_solve_batch(objective, contexts, jnp.asarray(cands), bounds, b, strategy="map")
        t0 = time.perf_counter()
        jax.block_until_ready(lbfgs_solve_batch(objective, contexts, jnp.asarray(cands), bounds,
                                                b, strategy="map")["fun"])
        batched_refine = (time.perf_counter() - t0) / n
        rows[str(seed)] = {"warm_forward_seconds": forward, "warm_refine_seconds": refine,
                           "refine_units": used, "cold_forward_seconds": cold_forward,
                           "cold_refine_seconds": cold_refine,
                           "batched_forward_seconds_per_instance": batched_forward,
                           "batched_refine_seconds_per_instance": batched_refine}
    result = {"family": name, "instances": n, "budget": b, "loss": chosen, "seeds": rows,
              "cpu_seconds": time.process_time() - c0}
    run_checkpoint.write_atomic(out / "result.json", json.dumps(result) + "\n")
    (out / "DONE").write_text("result.json\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify-splits")
    v.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("shard")
    s.add_argument("--family", choices=list(MATCHED_B), required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--start", type=int, required=True)
    s.add_argument("--stop", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("timing")
    t.add_argument("--family", choices=list(MATCHED_B), required=True)
    t.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-splits":
        return verify_splits(args.out)
    if args.command == "shard":
        if args.split not in ("validation", "test", *OOD_SPLITS[args.family]):
            raise SystemExit(f"unknown split {args.split}")
        return shard(args.family, args.split, args.start, args.stop, args.out)
    return timing(args.family, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
