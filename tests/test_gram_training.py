"""v0.7 infrastructure: matched Gram training unroll, its loss, and batched training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt.compiled import preconditioned_solve
from neuralopt.fast_training import TaskGroup, make_train_step, train_batched
from neuralopt.gram import HISTORY_V2_MASKS, GramConfig, init_gram_model
from neuralopt.gram_training import GramUnrollConfig, gram_regret_loss, gram_unroll

CONFIG = GramConfig(history_length=4, width=16, heads=4, layers=1, **HISTORY_V2_MASKS)


def _rotated_objective(d, seed=0, condition=100.0):
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(seed), (d, d)))
    matrix = q @ jnp.diag(jnp.geomspace(1.0, condition, d)) @ q.T
    center = 0.3 * jax.random.normal(jax.random.PRNGKey(seed + 1), (d,))

    def objective(x):
        return 0.5 * (x - center) @ matrix @ (x - center)

    return objective


@pytest.mark.parametrize("variant", ["full", "scalar", "nohist"])
def test_training_unroll_reproduces_the_compiled_engine(variant):
    extra = {"full": {}, "scalar": {"scalar_only": True}, "nohist": {"history_masked": True}}
    config = GramConfig(**{**CONFIG.__dict__, **extra[variant]})
    d, budget = 5, 40
    with jax.enable_x64(True):
        objective = _rotated_objective(d)
        lo, hi = -3.0 * jnp.ones(d), 3.0 * jnp.ones(d)
        x0 = jnp.full(d, 1.5)
        params = init_gram_model(jax.random.PRNGKey(2), config)
        rolled = gram_unroll(params, objective, (lo, hi), jax.random.PRNGKey(0), config,
                             GramUnrollConfig(budget=budget, horizons=(budget,)), x0=x0)
        engine = preconditioned_solve(objective, x0, (lo, hi), budget, params=params,
                                      config=config, model_kind="gram")
    active = np.asarray(rolled["active"])
    n = int(engine["iteration"])
    assert int(active.sum()) == n
    np.testing.assert_allclose(np.asarray(rolled["values"])[active],
                               np.asarray(engine["trajectory"][1 : n + 1]), rtol=1e-10)
    assert int(rolled["used"][-1]) == int(engine["used"])
    final = rolled["final_history"]
    for field in ("u", "g", "f"):
        np.testing.assert_allclose(np.asarray(getattr(final, field)),
                                   np.asarray(getattr(engine["history"], field)),
                                   rtol=1e-10, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(final.valid), np.asarray(engine["history"].valid))


def test_unroll_is_matched_to_the_evaluation_budget():
    """97 scan steps at budget 100: the most iterations the loop can ever run."""
    d = 2
    with jax.enable_x64(True):
        # The gradient pushes against the lower bound: every trial is rejected.
        objective = lambda x: jnp.sum(x)  # noqa: E731
        lo, hi = -jnp.ones(d), jnp.ones(d)
        params = init_gram_model(jax.random.PRNGKey(0), CONFIG)
        rolled = gram_unroll(params, objective, (lo, hi), jax.random.PRNGKey(0), CONFIG,
                             GramUnrollConfig(), x0=-jnp.ones(d))
    assert GramUnrollConfig().budget == 100
    assert np.asarray(rolled["active"]).shape == (97,)
    assert int(np.asarray(rolled["active"]).sum()) == 97
    assert int(rolled["used"][-1]) == 99  # one unit stays unspent, as in inference


@pytest.mark.parametrize("variant", ["full", "scalar", "nohist"])
def test_gram_loss_has_finite_nonzero_meta_gradient(variant):
    extra = {"full": {}, "scalar": {"scalar_only": True}, "nohist": {"history_masked": True}}
    config = GramConfig(**{**CONFIG.__dict__, **extra[variant]})
    d = 4
    with jax.enable_x64(True):
        objective = _rotated_objective(d, seed=3)
        bounds = (-3.0 * jnp.ones(d), 3.0 * jnp.ones(d))
        params = init_gram_model(jax.random.PRNGKey(1), config)
        unroll = GramUnrollConfig(budget=30, horizons=(10, 20, 30))
        loss, grads = jax.jit(jax.value_and_grad(
            lambda p: gram_regret_loss(p, objective, bounds, 0.0, jax.random.PRNGKey(4),
                                       config, unroll)))(params)
        leaves = jax.tree_util.tree_leaves(grads)
        assert bool(jnp.isfinite(loss))
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
        assert sum(float(jnp.sum(jnp.abs(g))) for g in leaves) > 0
        if variant == "scalar":  # A and C heads receive no gradient when A = C = 0
            for name in ("a_left", "a_right", "c_left", "c_right"):
                assert float(jnp.abs(grads[name]).sum()) == 0.0


def test_unroll_config_validation():
    with pytest.raises(ValueError):
        GramUnrollConfig(budget=2)
    with pytest.raises(ValueError):
        GramUnrollConfig(budget=50, horizons=(25, 100))


def _parametric_quadratic(x, theta):
    center, matrix = theta
    return 0.5 * (x - center) @ matrix @ (x - center)


def test_batched_gram_loss_with_one_task_equals_single_loss():
    d = 4
    with jax.enable_x64(True):
        q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(6), (d, d)))
        matrix = q @ jnp.diag(jnp.geomspace(1.0, 50.0, d)) @ q.T
        center = jnp.zeros(d)
        bounds = (-2.0 * jnp.ones(d), 2.0 * jnp.ones(d))
        params = init_gram_model(jax.random.PRNGKey(0), CONFIG)
        unroll = GramUnrollConfig(budget=30, horizons=(10, 30))
        key = jax.random.PRNGKey(8)
        single = gram_regret_loss(params, lambda x: _parametric_quadratic(x, (center, matrix)),
                                  bounds, 0.0, key, CONFIG, unroll)
        step = make_train_step(_parametric_quadratic, CONFIG, unroll, "gram", 1e-3)
        _, _, batched, skipped = step(params, step.init(params), (center[None], matrix[None]),
                                      bounds[0][None], bounds[1][None], jnp.zeros(1), key[None])
    assert not bool(skipped)
    np.testing.assert_allclose(float(batched), float(single), rtol=1e-10)


def _edge_singular(x, theta):
    del theta
    return jnp.sum(jnp.sqrt(1.0 + x))


def test_gram_training_skips_and_counts_nonfinite_meta_gradients():
    d, n = 1, 4
    config = GramConfig(**{**CONFIG.__dict__, "fallback_length": 1.0})
    group = TaskGroup(name="edge-singular-d1", objective=_edge_singular, thetas=jnp.zeros((n,)),
                      lo=-jnp.ones((n, d)), hi=jnp.ones((n, d)), optimum=jnp.zeros((n,)))
    unroll = GramUnrollConfig(budget=20, horizons=(10, 20))
    report = train_batched([group], seed=0, steps=12, batch_size=n, config=config,
                           unroll=unroll, model_kind="gram")
    assert report.skipped_updates >= 1 and len(report.losses) == 12
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in report.params.values())
    step = make_train_step(_edge_singular, config, unroll, "gram", 1e-3)
    params = init_gram_model(jax.random.PRNGKey(0), config)
    state = step.init(params)
    for attempt in range(50):
        keys = jax.random.split(jax.random.PRNGKey(attempt), n)
        new_params, new_state, _, skipped = step(params, state, group.thetas, group.lo,
                                                 group.hi, group.optimum, keys)
        if bool(skipped):
            break
    assert bool(skipped)
    for k in params:
        np.testing.assert_array_equal(new_params[k], params[k])
    for a, b in zip(jax.tree_util.tree_leaves(new_state), jax.tree_util.tree_leaves(state),
                    strict=True):
        np.testing.assert_array_equal(a, b)
