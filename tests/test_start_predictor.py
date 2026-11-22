"""v0.8 start predictor g_φ (pre-registration §5): shapes, box, invariances, losses,
checkpoint schema."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt.families import FAMILIES, family_objective, make_split
from neuralopt.start_predictor import (
    START_SCHEMA,
    StartConfig,
    candidate_diversity,
    context_stats,
    init_start_model,
    load_start_checkpoint,
    objective_loss,
    observation_tokens,
    predict_starts,
    regression_loss,
    save_start_checkpoint,
)


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


NAME = "cascade-realistic"


def _setup(n=6):
    contexts, truths = make_split(NAME, "validation", n)
    spec = FAMILIES[NAME]
    config = StartConfig(dimension=spec.dimension)
    params = init_start_model(jax.random.PRNGKey(0), config)
    stats = context_stats(contexts)
    one = jax.tree_util.tree_map(lambda a: a[0], contexts)
    return contexts, truths, spec, config, params, stats, one


def test_tokens_hold_time_value_species_noise_and_dose_with_padding_zeroed():
    contexts, _, _, _, _, stats, one = _setup()
    tokens = observation_tokens(one, stats)
    assert tokens.shape == (one["t"].shape[0], 6)
    mask = np.asarray(one["mask"])
    np.testing.assert_allclose(tokens[~mask], 0.0)
    np.testing.assert_allclose(tokens[mask, 0], np.asarray(one["t"])[mask] / 10)
    species = np.asarray(one["species"])[mask]
    np.testing.assert_allclose(tokens[mask, 2], (species == 0).astype(float))
    np.testing.assert_allclose(tokens[mask, 3], (species == 1).astype(float))
    np.testing.assert_allclose(tokens[mask, 4], np.log(np.asarray(one["sigma"])[mask]))
    np.testing.assert_allclose(tokens[mask, 5], float(one["dose"]))
    y = np.asarray(one["y"])[mask]
    expected = (y - np.asarray(stats["y_mean"])[species]) / np.asarray(stats["y_std"])[species]
    np.testing.assert_allclose(tokens[mask, 1], expected)


def test_candidates_lie_strictly_inside_the_box_and_confidences_are_probabilities():
    _, _, spec, config, params, stats, one = _setup()
    out = predict_starts(params, one, stats, spec.lo, spec.hi, config)
    c = np.asarray(out["candidates"])
    assert c.shape == (config.candidates, spec.dimension)
    assert np.all(c > spec.lo) and np.all(c < spec.hi)
    p = np.asarray(out["confidence"])
    assert p.shape == (config.candidates,) and np.all((p > 0) & (p < 1))
    order = np.asarray(out["order"])
    assert np.all(np.diff(p[order]) <= 0)  # sorted by confidence, highest first


def test_outputs_ignore_observation_order_and_masked_padding():
    _, _, spec, config, params, stats, one = _setup()
    base = predict_starts(params, one, stats, spec.lo, spec.hi, config)
    n = one["t"].shape[0]
    perm = np.random.default_rng(0).permutation(n)
    shuffled = {k: (v[perm] if getattr(v, "ndim", 0) == 1 and v.shape[0] == n else v)
                for k, v in one.items()}
    moved = predict_starts(params, shuffled, stats, spec.lo, spec.hi, config)
    np.testing.assert_allclose(moved["candidates"], base["candidates"], atol=1e-10)
    np.testing.assert_allclose(moved["confidence"], base["confidence"], atol=1e-10)
    garbage = dict(one)
    pad = ~np.asarray(one["mask"])
    garbage["y"] = jnp.where(pad, 1e6, one["y"])
    garbage["t"] = jnp.where(pad, 99.0, one["t"])
    same = predict_starts(params, garbage, stats, spec.lo, spec.hi, config)
    np.testing.assert_allclose(same["candidates"], base["candidates"], atol=1e-10)


def test_regression_and_objective_losses_are_best_of_k():
    _, truths, spec, config, _, _, one = _setup()
    lo, hi = jnp.asarray(spec.lo), jnp.asarray(spec.hi)
    target = truths[0]
    cands = jnp.stack([target + 0.3, target, target - 0.2])
    assert float(regression_loss(cands, target, lo, hi)) == pytest.approx(0.0, abs=1e-14)
    far = jnp.stack([target + 0.3, target - 0.2])
    u = lambda x: 2 * (x - lo) / (hi - lo) - 1  # noqa: E731
    expected = min(float(jnp.sum((u(c) - u(target)) ** 2)) for c in far)
    assert float(regression_loss(far, target, lo, hi)) == pytest.approx(expected)
    f = family_objective(NAME)
    values = [float(f(c, one)) for c in cands]
    assert float(objective_loss(cands, f, one, 2.0)) == pytest.approx(min(values) / 2.0)


def test_losses_have_finite_nonzero_parameter_gradients():
    _, truths, spec, config, params, stats, one = _setup()
    lo, hi = jnp.asarray(spec.lo), jnp.asarray(spec.hi)
    f = family_objective(NAME)

    def loss(p):
        c = predict_starts(p, one, stats, lo, hi, config)["candidates"]
        return regression_loss(c, truths[0], lo, hi) + objective_loss(c, f, one, 1.0)

    grads = jax.grad(loss)(params)
    flat = jnp.concatenate([jnp.ravel(g) for g in jax.tree_util.tree_leaves(grads)])
    assert bool(jnp.all(jnp.isfinite(flat))) and float(jnp.max(jnp.abs(flat))) > 0


def test_diversity_is_zero_for_collapsed_candidates():
    _, truths, spec, _, _, _, _ = _setup()
    collapsed = jnp.tile(truths[0], (8, 1))
    assert float(candidate_diversity(collapsed, spec.lo, spec.hi)) == 0.0
    spread = collapsed + jnp.linspace(0, 1, 8)[:, None]
    assert float(candidate_diversity(spread, spec.lo, spec.hi)) > 0.0


def test_checkpoint_round_trip_and_fail_closed(tmp_path):
    _, _, spec, config, params, stats, one = _setup()
    path = tmp_path / "start.npz"
    save_start_checkpoint(path, params, config, stats, {"family": NAME})
    loaded, loaded_config, loaded_stats = load_start_checkpoint(path)
    assert loaded_config == config
    a = predict_starts(params, one, stats, spec.lo, spec.hi, config)
    b = predict_starts(loaded, one, loaded_stats, spec.lo, spec.hi, loaded_config)
    np.testing.assert_allclose(a["candidates"], b["candidates"])
    with np.load(path) as data:
        arrays = dict(data)
    arrays["schema"] = np.array(["neuralopt-start/v0.0"])
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="schema"):
        load_start_checkpoint(tmp_path / "bad.npz")
    arrays["schema"] = np.array([START_SCHEMA])
    arrays["param__embed"] = arrays["param__embed"][:, :3]
    np.savez(tmp_path / "shape.npz", **arrays)
    with pytest.raises(ValueError, match="match"):
        load_start_checkpoint(tmp_path / "shape.npz")
