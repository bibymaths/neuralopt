"""EXPLORATORY POST-MORTEM for v0.7 (not pre-registered, no new claims, feeds no decision).

Hypothesis tested: the model configuration (depth, width) limits learning A and C.
Training and validation tasks only; test tasks are never built. Frozen v0.7 results and
checkpoints are not touched.

1. Imitation sweep. Oracle-BFGS runs (frozen halving loop, K = 10, history-v2 masks,
   float64, budget 100) on the 96 TRAINING tasks give token/target pairs. gram-full
   networks with layers in {2, 4, 6} x width in {32, 64, 128} (heads = width / 8, i.e.
   head dimension 8 as in the frozen model) are fitted by supervised regression to the
   oracle's A, C C^T (C is only defined up to a right orthogonal factor), delta and
   sigma = 1. Reported: parameter count; fit error on held-out VALIDATION histories
   (oracle runs on validation tasks, seeds 8500+); and median log10 regret when the
   fitted model runs the shared loop on validation tasks, next to the oracle, BB and the
   frozen trained gram-full checkpoints.
2. Head saddle check. 200 updates of the frozen meta-training (training tasks; same
   batching as train_batched) for gram-full, with the frozen init (A/C head weights at
   0.1 scale) and with A/C head weights rescaled to 1.0 scale; per-update gradient norms
   reaching a_left/a_right/c_left/c_right vs delta/sigma, and A/C output magnitudes on
   fixed validation probe states.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ["JAX_ENABLE_X64"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from neuralopt.compiled import preconditioned_solve  # noqa: E402
from neuralopt.gram import (  # noqa: E402
    GramConfig,
    RejectionInfo,
    apply_gram_model,
    apply_inverse_metric,
    gram_basis,
    gram_tokens,
    init_gram_model,
    tau_scale,
)
from neuralopt.gram_training import gram_regret_loss  # noqa: E402
from neuralopt.task_families import family_objective  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gram_space_v0_7 as protocol  # noqa: E402
from run_gram_gates_v0_7 import oracle_coefficients  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".superpowers" / "postmortem-v0.7"  # git-ignored derived data
RESULTS = ROOT / "results" / "gram-postmortem-v0.7.0.json"
CONFIG = protocol.REFERENCE_CONFIG  # K = 10, history-v2 masks (tokens are model-independent)
SWEEP = [(layers, width) for layers in (2, 4, 6) for width in (32, 64, 128)]
FIT_STEPS, FIT_BATCH, FIT_LR = 3000, 256, 1e-3
VALIDATION_DIMS, VALIDATION_REPS = (6, 10), 2
SADDLE_UPDATES, SADDLE_SEEDS, PROBE_EVERY = 200, (0, 1, 2), 25
AC_HEADS = ("a_left", "a_right", "c_left", "c_right")
if os.environ.get("POSTMORTEM_SMOKE"):  # quick end-to-end check of the pipeline only
    SWEEP, FIT_STEPS, SADDLE_UPDATES, SADDLE_SEEDS, PROBE_EVERY = SWEEP[:2], 20, 4, (0,), 2
    RESULTS = CACHE / "smoke.json"


def sweep_config(layers: int, width: int) -> GramConfig:
    return GramConfig(**{**protocol.BASE, "layers": layers, "width": width,
                         "heads": width // 8})


def validation_specs() -> list[dict[str, Any]]:
    """Seeds 8500+ (disjoint from training 7000+, the d=32 ceiling seeds 8000-8303 and
    test 9000+); dimensions 6 and 10 (neither training nor test dimensions)."""
    specs = []
    for i, (family, condition) in enumerate(protocol.TRAIN_VARIANTS):
        for j, dim in enumerate(VALIDATION_DIMS):
            for r in range(VALIDATION_REPS):
                a, c = protocol._scales(i + j + r)
                specs.append(dict(family=family, condition=condition, dimension=dim,
                                  seed=8500 + 100 * i + 10 * j + r, objective_scale=a,
                                  coordinate_scale=c, replicate=r))
    return specs


# ---- oracle state collection --------------------------------------------------------------
def collect(specs: list[dict[str, Any]], keep_geometry: bool) -> dict[str, Any]:
    tokens, masks, targets_a, targets_cc, targets_delta, geometry = [], [], [], [], [], []

    def per_state(hist: Any, g_u: Any, used: Any, rl: Any, rr: Any, rh: Any) -> Any:
        rejection = RejectionInfo(hist.rejections, rl, rr, rh)
        tok = gram_tokens(hist, g_u, rejection, 1 - used / protocol.PRIMARY_BUDGET, CONFIG)
        basis = gram_basis(hist, CONFIG)
        tau = tau_scale(basis, g_u, CONFIG.fallback_length)
        A, C, delta, _ = oracle_coefficients(basis, g_u, tau)
        return tok, basis.mask, A, C @ C.T, delta

    batched = jax.jit(jax.vmap(per_state))
    for spec in specs:
        task = protocol.build(spec)
        run = protocol.reference_loop("oracle", task.family, protocol.PRIMARY_BUDGET,
                                      collect=protocol.PRIMARY_BUDGET)
        out = run(task.theta, task.x0, task.lo, task.hi)
        n = int(out["iterations"])
        st = jax.tree_util.tree_map(lambda a, n=n: a[:n], out["states"])
        tok, mask, A, CC, delta = batched(st["hist"], st["g_u"], st["used"], st["rl"],
                                          st["rr"], st["rh"])
        tokens.append(np.asarray(tok))
        masks.append(np.asarray(mask))
        targets_a.append(np.asarray(A))
        targets_cc.append(np.asarray(CC))
        targets_delta.append(np.asarray(delta))
        if keep_geometry:
            geometry.append({"spec": spec, "states": st})
    data = {k: np.concatenate(v) for k, v in (("tokens", tokens), ("mask", masks),
                                              ("A", targets_a), ("CC", targets_cc),
                                              ("delta", targets_delta))}
    data["geometry"] = geometry
    return data


# ---- imitation sweep ------------------------------------------------------------------------
def regression_loss(params: Any, batch: dict[str, Any], config: GramConfig) -> Any:
    def one(tok: Any, mask: Any) -> Any:
        out = apply_gram_model(params, tok, mask, config)
        return out["A"], out["C"] @ out["C"].T, out["delta"], out["sigma"]

    A, CC, delta, sigma = jax.vmap(one)(batch["tokens"], batch["mask"])
    error_a = jnp.sum((A - batch["A"]) ** 2, axis=(1, 2)) / (
        jnp.sum(batch["A"] ** 2, axis=(1, 2)) + 1.0)
    error_c = jnp.sum((CC - batch["CC"]) ** 2, axis=(1, 2)) / (
        jnp.sum(batch["CC"] ** 2, axis=(1, 2)) + 1.0)
    return jnp.mean(error_a + error_c + (delta - batch["delta"]) ** 2 + jnp.log(sigma) ** 2)


def fit_errors(params: Any, data: dict[str, Any], config: GramConfig) -> dict[str, float]:
    def one(tok: Any, mask: Any) -> Any:
        out = apply_gram_model(params, tok, mask, config)
        return out["A"], out["C"] @ out["C"].T, out["delta"], out["sigma"]

    A, CC, delta, sigma = jax.jit(jax.vmap(one))(jnp.asarray(data["tokens"]),
                                                jnp.asarray(data["mask"]))
    rel_a = np.sqrt(np.sum((np.asarray(A) - data["A"]) ** 2, axis=(1, 2))
                    / (np.sum(data["A"] ** 2, axis=(1, 2)) + 1e-12))
    rel_c = np.sqrt(np.sum((np.asarray(CC) - data["CC"]) ** 2, axis=(1, 2))
                    / (np.sum(data["CC"] ** 2, axis=(1, 2)) + 1e-12))
    has_pairs = data["mask"].any(axis=1)
    cosines = []
    for item in data["geometry"]:
        st = item["states"]

        def direction(hist: Any, g_u: Any, used: Any, rl: Any, rr: Any, rh: Any) -> Any:
            rejection = RejectionInfo(hist.rejections, rl, rr, rh)
            tok = gram_tokens(hist, g_u, rejection, 1 - used / protocol.PRIMARY_BUDGET,
                              CONFIG)
            basis = gram_basis(hist, CONFIG)
            out = apply_gram_model(params, tok, basis.mask, config)
            tau = tau_scale(basis, g_u, CONFIG.fallback_length)
            return out["sigma"] * apply_inverse_metric(g_u, basis, out["A"], out["C"],
                                                       out["delta"], tau, CONFIG.eps_rel)

        model = np.asarray(jax.jit(jax.vmap(direction))(st["hist"], st["g_u"], st["used"],
                                                         st["rl"], st["rr"], st["rh"]))
        oracle = np.asarray(st["oracle"])
        cosines.extend(np.sum(model * oracle, axis=1)
                       / (np.linalg.norm(model, axis=1) * np.linalg.norm(oracle, axis=1)
                          + 1e-300))
    return {
        "median_relative_error_A": float(np.median(rel_a[has_pairs])),
        "median_relative_error_CCt": float(np.median(rel_c[has_pairs])),
        "median_abs_error_delta": float(np.median(np.abs(np.asarray(delta) - data["delta"]))),
        "median_abs_log_sigma": float(np.median(np.abs(np.log(np.asarray(sigma))))),
        "median_direction_cosine_with_oracle": float(np.median(cosines)),
        "states": int(len(rel_a)), "states_with_pairs": int(has_pairs.sum()),
    }


def validation_regret(params: Any, config: GramConfig, specs: list[dict[str, Any]]) -> list[float]:
    values = []
    for spec in specs:
        task = protocol.build(spec)
        objective = family_objective(task.family)
        out = preconditioned_solve(objective, task.x0, (task.lo, task.hi),
                                   protocol.PRIMARY_BUDGET, theta=task.theta, params=params,
                                   config=config, model_kind="gram")
        f0 = float(objective(task.x0, task.theta))
        values.append(protocol._log10_regret(float(out["value"]), f0, 0.0))
    return values


def run_sweep_config(job: dict[str, Any]) -> dict[str, Any]:
    layers, width = job["layers"], job["width"]
    import optax

    config = sweep_config(layers, width)
    train = np.load(CACHE / "train.npz")
    train = {k: train[k] for k in train.files}
    validation = _load_validation()
    params = init_gram_model(jax.random.PRNGKey(0), config)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(FIT_LR))
    state = optimizer.init(params)

    @jax.jit
    def step(params: Any, state: Any, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
        loss, grads = jax.value_and_grad(regression_loss)(params, batch, config)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    rng = np.random.default_rng(0)
    n = train["tokens"].shape[0]
    start = time.perf_counter()
    losses = []
    for _ in range(FIT_STEPS):
        index = rng.integers(0, n, FIT_BATCH)
        batch = {k: jnp.asarray(train[k][index]) for k in ("tokens", "mask", "A", "CC",
                                                            "delta")}
        params, state, loss = step(params, state, batch)
        losses.append(float(loss))
    seconds = time.perf_counter() - start
    curvature = [s for s in validation_specs() if protocol.build(s).label in protocol.CURVATURE]
    regret = validation_regret(params, config, curvature)
    return {
        "layers": layers, "width": width, "heads": width // 8,
        "parameter_count": int(sum(np.asarray(v).size for v in params.values())),
        "fit_seconds": seconds, "final_train_loss_mean_last_100": float(np.mean(losses[-100:])),
        "validation_fit": fit_errors(params, validation, config),
        "validation_curvature_log10_regret": regret,
        "validation_curvature_median_log10_regret": float(np.median(regret)),
    }


def _load_validation() -> dict[str, Any]:
    specs = validation_specs()
    return collect(specs, keep_geometry=True)


# ---- references on validation tasks -------------------------------------------------------
def validation_references() -> dict[str, Any]:
    curvature = [s for s in validation_specs() if protocol.build(s).label in protocol.CURVATURE]
    out: dict[str, Any] = {}
    for source in ("bb", "oracle"):
        values = []
        for spec in curvature:
            task = protocol.build(spec)
            res = protocol.reference_loop(source, task.family, protocol.PRIMARY_BUDGET)(
                task.theta, task.x0, task.lo, task.hi)
            f0 = float(family_objective(task.family)(task.x0, task.theta))
            values.append(protocol._log10_regret(float(res["value"]), f0, 0.0))
        out[source] = {"log10_regret": values, "median": float(np.median(values))}
    frozen = []
    for _seed, params in protocol.load_arm("gram-full"):
        frozen.append(float(np.median(validation_regret(params, protocol.ARMS["gram-full"],
                                                        curvature))))
    out["frozen_gram_full_trained"] = {"median_per_seed": frozen,
                                       "median": float(np.median(frozen))}
    out["tasks"] = len(curvature)
    return out


# ---- head saddle check ----------------------------------------------------------------------
def run_saddle(job: dict[str, Any]) -> dict[str, Any]:
    import optax

    variant, seed = job["variant"], job["seed"]
    config = protocol.ARMS["gram-full"]
    groups = protocol.training_groups()
    probe = np.load(CACHE / "probe.npz")
    probe_tokens, probe_mask = jnp.asarray(probe["tokens"]), jnp.asarray(probe["mask"])
    key = jax.random.PRNGKey(seed)
    params = init_gram_model(key, config)
    if variant == "init-1.0":
        params = {k: v * (10.0 if k in AC_HEADS else 1.0) for k, v in params.items()}
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(protocol.LEARNING_RATE))
    state = optimizer.init(params)
    names = {"A_heads": ("a_left", "a_right"), "C_heads": ("c_left", "c_right"),
             "delta": ("delta",), "sigma": ("sigma",)}

    def make_step(objective: Any) -> Any:
        @jax.jit
        def step(params: Any, state: Any, thetas: Any, lo: Any, hi: Any, optimum: Any,
                 keys: Any) -> tuple[Any, ...]:
            def loss_fn(p: Any) -> Any:
                return jnp.mean(jax.vmap(lambda th, lo_i, hi_i, o, k: gram_regret_loss(
                    p, lambda x: objective(x, th), (lo_i, hi_i), o, k, config, protocol.UNROLL))(
                    thetas, lo, hi, optimum, keys))

            loss, grads = jax.value_and_grad(loss_fn)(params)
            norms = {g: jnp.sqrt(sum(jnp.sum(grads[k] ** 2) for k in ks))
                     for g, ks in names.items()}
            norms["total"] = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            finite = jnp.isfinite(loss) & jnp.all(jnp.asarray(
                [jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
            safe = jax.tree_util.tree_map(lambda g: jnp.where(finite, g, 0.0), grads)
            updates, new_state = optimizer.update(safe, state, params)
            new_params = optax.apply_updates(params, updates)
            keep = lambda n, o: jax.tree_util.tree_map(  # noqa: E731
                lambda a, b: jnp.where(finite, a, b), n, o)
            return keep(new_params, params), keep(new_state, state), loss, norms, ~finite

        return step

    steps = {g.name: make_step(g.objective) for g in groups}

    @jax.jit
    def magnitudes(params: Any) -> Any:
        out = jax.vmap(lambda t, m: apply_gram_model(params, t, m, config))(probe_tokens,
                                                                             probe_mask)
        return {"A_fro": jnp.mean(jnp.sqrt(jnp.sum(out["A"] ** 2, axis=(1, 2)))),
                "C_fro": jnp.mean(jnp.sqrt(jnp.sum(out["C"] ** 2, axis=(1, 2)))),
                "abs_delta": jnp.mean(jnp.abs(out["delta"])),
                "sigma": jnp.mean(out["sigma"])}

    trace, mags = [], [{"update": 0, **{k: float(v) for k, v in magnitudes(params).items()}}]
    for update in range(SADDLE_UPDATES):  # the same batching as train_batched
        group = groups[update % len(groups)]
        key, batch_key, loss_key = jax.random.split(key, 3)
        size = min(protocol.BATCH, group.size)
        index = np.asarray(jax.random.choice(batch_key, group.size, (size,), replace=False))
        batch = jax.tree_util.tree_map(lambda a, i=index: a[i], group.thetas)
        params, state, loss, norms, skipped = steps[group.name](
            params, state, batch, group.lo[index], group.hi[index], group.optimum[index],
            jax.random.split(loss_key, size))
        trace.append({"loss": float(loss), "skipped": bool(skipped),
                      **{k: float(v) for k, v in norms.items()}})
        if (update + 1) % PROBE_EVERY == 0:
            mags.append({"update": update + 1,
                         **{k: float(v) for k, v in magnitudes(params).items()}})
    replay = None
    if variant == "init-0.1" and seed == SADDLE_SEEDS[0]:
        # The saddle loop must reproduce the frozen trainer's first updates exactly.
        from neuralopt.fast_training import train_batched

        reference = train_batched(groups, seed=seed, steps=SADDLE_UPDATES,
                                  batch_size=protocol.BATCH,
                                  learning_rate=protocol.LEARNING_RATE, config=config,
                                  unroll=protocol.UNROLL, model_kind="gram").params
        replay = float(max(np.max(np.abs(np.asarray(reference[k]) - np.asarray(params[k])))
                           for k in params))
    arr = {k: np.array([t[k] for t in trace]) for k in ("A_heads", "C_heads", "delta", "sigma",
                                                         "total")}
    return {
        "variant": variant, "seed": seed, "updates": SADDLE_UPDATES,
        "max_abs_param_difference_vs_train_batched": replay,
        "skipped": int(sum(t["skipped"] for t in trace)),
        "median_grad_norm": {k: float(np.median(v)) for k, v in arr.items()},
        "median_grad_norm_first_50": {k: float(np.median(v[:50])) for k, v in arr.items()},
        "median_grad_norm_last_50": {k: float(np.median(v[-50:])) for k, v in arr.items()},
        "median_ratio_AC_to_delta_sigma": float(np.median(
            np.sqrt(arr["A_heads"] ** 2 + arr["C_heads"] ** 2)
            / np.sqrt(arr["delta"] ** 2 + arr["sigma"] ** 2 + 1e-300))),
        "output_magnitudes": mags,
        "per_update": trace,
    }


# ---- orchestration ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    prereg = protocol.check_preregistration()
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / ".gitignore").write_text("*\n")  # derived data, never committed
    start = time.perf_counter()
    train = collect(protocol.training_specs(), keep_geometry=False)
    np.savez(CACHE / "train.npz", **{k: v for k, v in train.items() if k != "geometry"})
    validation = collect(validation_specs(), keep_geometry=False)
    rng = np.random.default_rng(0)
    probe_index = rng.choice(validation["tokens"].shape[0], 256, replace=False)
    np.savez(CACHE / "probe.npz", tokens=validation["tokens"][probe_index],
             mask=validation["mask"][probe_index])
    collect_seconds = time.perf_counter() - start
    references = validation_references()
    jobs = [{"layers": layer, "width": width} for layer, width in SWEEP]
    saddle_jobs = [{"variant": v, "seed": s} for v in ("init-0.1", "init-1.0")
                   for s in SADDLE_SEEDS]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        sweep_future = pool.map(run_sweep_config, jobs)
        saddle_future = pool.map(run_saddle, saddle_jobs)
        sweep, saddle = list(sweep_future), list(saddle_future)
    output = {
        "label": "EXPLORATORY POST-MORTEM - not pre-registered - no claims - training and "
        "validation tasks only (no test task built)",
        "preregistration_sha256": prereg,
        "train_split_sha256": protocol.split_hash(protocol.training_specs()),
        "validation_split_sha256": protocol.split_hash(validation_specs()),
        "oracle_states": {"train": int(train["tokens"].shape[0]),
                          "validation": int(validation["tokens"].shape[0]),
                          "collect_seconds": collect_seconds},
        "fit": {"steps": FIT_STEPS, "batch": FIT_BATCH, "learning_rate": FIT_LR,
                "loss": "rel. Frobenius A + rel. Frobenius CC^T + (delta err)^2 + "
                "(log sigma)^2; seed 0"},
        "validation_references": references,
        "imitation_sweep": sweep,
        "head_saddle": saddle,
    }
    RESULTS.write_text(json.dumps(output, indent=1))
    print(json.dumps({k: v for k, v in output.items() if k != "head_saddle"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
