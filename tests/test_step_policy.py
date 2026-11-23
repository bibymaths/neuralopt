"""v0.8-B learned step policy (plan docs/plans/neuralopt-v08b-step-policy.md §3.1):
shapes, size, coordinate equivariance, masks, box, accounting, checkpoint schema."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt.families import SYNTHETIC, family_objective, make_split
from neuralopt.step_policy import (
    POLICY_SCHEMA,
    PolicyConfig,
    empty_window,
    init_policy,
    load_policy_checkpoint,
    parameter_count,
    policy_outputs,
    policy_rollout,
    push_window,
    save_policy_checkpoint,
)


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


def _window(d=4, filled=5, seed=0):
    config = PolicyConfig()
    rng = np.random.default_rng(seed)
    w = empty_window(d, config)
    for _ in range(filled):
        w = push_window(w, jnp.asarray(rng.uniform(-1, 1, d)), jnp.asarray(rng.normal(size=d)),
                        float(rng.normal()))
    return config, w


def test_parameter_count_is_in_the_planned_range():
    config = PolicyConfig()
    assert 20_000 <= parameter_count(init_policy(jax.random.PRNGKey(0), config)) <= 50_000
    with_context = PolicyConfig(context=True)
    assert parameter_count(init_policy(jax.random.PRNGKey(0), with_context)) <= 50_000


def test_outputs_are_per_coordinate_and_bounded():
    config, w = _window()
    params = init_policy(jax.random.PRNGKey(0), config)
    mu, log_sigma = policy_outputs(params, w, budget_fraction=0.5, config=config)
    assert mu.shape == (4,) and log_sigma.shape == (4,)
    lo, hi = np.log(config.sigma_min), np.log(config.sigma_max)
    assert np.all((np.asarray(log_sigma) >= lo - 1e-12) & (np.asarray(log_sigma) <= hi + 1e-12))
    assert np.all(np.abs(np.asarray(mu)) <= config.log_step_bound + 1e-12)


def test_outputs_permute_with_the_coordinates():
    config, w = _window()
    params = init_policy(jax.random.PRNGKey(1), config)
    perm = np.array([2, 0, 3, 1])
    permuted = w._replace(u=w.u[:, perm], g=w.g[:, perm])
    mu, ls = policy_outputs(params, w, 0.3, config)
    mu_p, ls_p = policy_outputs(params, permuted, 0.3, config)
    np.testing.assert_allclose(mu_p, mu[perm], atol=1e-10)
    np.testing.assert_allclose(ls_p, ls[perm], atol=1e-10)


def test_unfilled_history_slots_do_not_matter():
    config, w = _window(filled=3)
    params = init_policy(jax.random.PRNGKey(2), config)
    garbage = w._replace(u=jnp.where(w.mask[:, None], w.u, 7.0),
                         g=jnp.where(w.mask[:, None], w.g, -9.0),
                         f=jnp.where(w.mask, w.f, 123.0))
    a = policy_outputs(params, w, 0.2, config)
    b = policy_outputs(params, garbage, 0.2, config)
    np.testing.assert_allclose(a[0], b[0], atol=1e-10)
    np.testing.assert_allclose(a[1], b[1], atol=1e-10)


def test_outputs_do_not_depend_on_the_number_of_empty_slots():
    # The same three states in a 16-slot and an 8-slot window: empty slots must not dilute
    # attention (no positional encoding, so the parameters do not depend on the history).
    long_config, w_long = _window(filled=3)
    short_config = PolicyConfig(history=8)
    rng = np.random.default_rng(0)
    w_short = empty_window(4, short_config)
    for _ in range(3):
        w_short = push_window(w_short, jnp.asarray(rng.uniform(-1, 1, 4)),
                              jnp.asarray(rng.normal(size=4)), float(rng.normal()))
    params = init_policy(jax.random.PRNGKey(4), long_config)
    a = policy_outputs(params, w_long, 0.4, long_config)
    b = policy_outputs(params, w_short, 0.4, short_config)
    np.testing.assert_allclose(a[0], b[0], atol=1e-10)
    np.testing.assert_allclose(a[1], b[1], atol=1e-10)


@pytest.mark.parametrize("budget", [20, 21, 100])
def test_rollout_spends_two_units_per_point_stays_in_the_box_and_starts_at_x0(budget):
    name = "rastrigin"
    contexts, _ = make_split(name, "validation", 1)
    c = jax.tree_util.tree_map(lambda a: a[0], contexts)
    lo, hi = (jnp.asarray(b) for b in SYNTHETIC[name].bounds("validation"))
    config = PolicyConfig()
    params = init_policy(jax.random.PRNGKey(3), config)
    out = policy_rollout(params, family_objective(name), c, c["x0"], lo, hi, budget,
                         jax.random.PRNGKey(7), config)
    points = budget // 2
    assert out["units"].shape == (points,)
    np.testing.assert_array_equal(out["units"], 2 * np.arange(1, points + 1))
    assert out["objective_calls"] == out["gradient_calls"] == points
    np.testing.assert_allclose(out["points"][0], c["x0"])
    assert np.all((np.asarray(out["points"]) >= lo) & (np.asarray(out["points"]) <= hi))
    f = family_objective(name)
    np.testing.assert_allclose(out["values"][3], f(out["points"][3], c), rtol=1e-12)
    again = policy_rollout(params, f, c, c["x0"], lo, hi, budget, jax.random.PRNGKey(7), config)
    np.testing.assert_array_equal(again["values"], out["values"])


def test_checkpoint_round_trip_and_fail_closed(tmp_path):
    config = PolicyConfig()
    params = init_policy(jax.random.PRNGKey(0), config)
    path = tmp_path / "policy.npz"
    save_policy_checkpoint(path, params, config, {"family": "rastrigin"})
    loaded, loaded_config = load_policy_checkpoint(path)
    assert loaded_config == config
    with np.load(path) as data:
        arrays = dict(data)
    arrays["schema"] = np.array(["step-policy-v0"])
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="schema"):
        load_policy_checkpoint(tmp_path / "bad.npz")
    assert POLICY_SCHEMA == "step-policy-v1"
