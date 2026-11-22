"""Track A5: parametric task families and the compiled, batched, NaN-guarded trainer."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import Problem
from neuralopt.fast_training import TaskGroup, make_train_step, train_batched
from neuralopt.model import ModelConfig, init_model
from neuralopt.task_families import family_objective, make_task, stack_tasks
from neuralopt.tasks import analytic_task, quadratic_task
from neuralopt.training import HistoryUnrollConfig, history_regret_loss

CONFIG = ModelConfig(
    width=16, heads=4, rank=2, layers=1, scale_aware=True, geometry_bound=10.0,
    lowrank_scale=0.5, feature_set="history-v1", history_length=4,
)
UNROLL = HistoryUnrollConfig(steps=6, horizons=(3, 6))


def _reference(family, condition, dimension, seed, a, c):
    """The v0.6 harness generator (scaled_task), rebuilt here as the oracle."""
    if family in ("isotropic", "diagonal", "rotated"):
        base = quadratic_task(jax.random.PRNGKey(seed), dimension, condition,
                              rotated=family == "rotated")
    else:
        base = analytic_task(family, dimension, seed)
    lo, hi = base.bounds
    return (lambda x: a * base.objective(x / c)), (lo * c, hi * c), base.x0 * c


@pytest.mark.parametrize(
    ("family", "condition"),
    [("isotropic", 1.0), ("diagonal", 1000.0), ("rotated", 100.0), ("rastrigin", 0.0),
     ("ackley", 0.0), ("rosenbrock", 0.0)],
)
def test_parametric_families_reproduce_the_v06_generator(family, condition):
    with jax.enable_x64(True):
        task = make_task(family, condition, 5, 1234, objective_scale=5.0, coordinate_scale=0.5)
        reference, (lo, hi), x0 = _reference(family, condition, 5, 1234, 5.0, 0.5)
        np.testing.assert_allclose(task.lo, lo)
        np.testing.assert_allclose(task.hi, hi)
        np.testing.assert_allclose(task.x0, x0)
        objective = family_objective(task.family)
        xs = jax.random.uniform(jax.random.PRNGKey(0), (8, 5), minval=lo, maxval=hi)
        for x in xs:
            np.testing.assert_allclose(objective(x, task.theta), reference(x), rtol=1e-10)
        assert task.known_optimum == 0.0


def test_batched_loss_with_one_task_equals_the_single_task_loss():
    with jax.enable_x64(True):
        task = make_task("rotated", 100.0, 4, 77, objective_scale=0.2, coordinate_scale=2.0)
        objective = family_objective(task.family)
        problem = Problem(lambda x: objective(x, task.theta), (task.lo, task.hi), 4,
                          dtype=jnp.float64, metadata={"known_optimum": 0.0})
        params = init_model(jax.random.PRNGKey(0), CONFIG)
        key = jax.random.PRNGKey(5)
        single = history_regret_loss(params, problem, key, CONFIG, UNROLL)
        group = TaskGroup.from_tasks([task])
        step = make_train_step(group.objective, CONFIG, UNROLL, "transformer", 1e-3)
        state = step.init(params)
        _, _, batched, skipped = step(params, state, group.thetas, group.lo, group.hi,
                                      group.optimum, key[None])
    assert not bool(skipped)
    np.testing.assert_allclose(float(batched), float(single), rtol=1e-10)


def test_train_step_compiles_once_per_family_and_dimension():
    tasks = [make_task("rotated", 100.0, 3, s, 1.0, 1.0) for s in range(4)]
    tasks += [make_task("rastrigin", 0.0, 3, s, 1.0, 1.0) for s in range(4)]
    groups = stack_tasks(tasks)
    assert {g.name for g in groups} == {"quadratic-d3", "rastrigin-d3"}
    train_batched(groups, steps=2, batch_size=2, config=CONFIG, unroll=UNROLL)
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture()
    logging.getLogger("jax").addHandler(handler)
    try:
        with jax.log_compiles(True):
            report = train_batched(groups, steps=4, batch_size=2, config=CONFIG, unroll=UNROLL)
    finally:
        logging.getLogger("jax").removeHandler(handler)
    assert report.compiled_groups == 2
    assert not [m for m in messages if "Compiling" in m and "train_step" in m]


def _edge_singular(x, theta):
    """Minimum 0 at the lower box edge, where the gradient is infinite."""
    del theta
    return jnp.sum(jnp.sqrt(1.0 + x))


def test_nonfinite_meta_gradients_skip_the_update_and_are_counted():
    d, n = 1, 4
    group = TaskGroup(
        name="edge-singular-d1",
        objective=_edge_singular,
        thetas=jnp.zeros((n,)),
        lo=-jnp.ones((n, d)),
        hi=jnp.ones((n, d)),
        optimum=jnp.zeros((n,)),
    )
    report = train_batched([group], seed=0, steps=12, batch_size=n, config=CONFIG,
                           unroll=HistoryUnrollConfig(steps=8, horizons=(8,)))
    assert report.skipped_updates >= 1
    assert len(report.skipped_steps) == report.skipped_updates
    assert len(report.losses) == 12  # every step reported, skipped or not
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in report.params.values())
    # A skipped update leaves parameters and optimizer state exactly unchanged.
    step = make_train_step(_edge_singular, CONFIG, HistoryUnrollConfig(steps=8, horizons=(8,)),
                           "transformer", 1e-3)
    params = init_model(jax.random.PRNGKey(0), CONFIG)
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


def test_shifted_rosenbrock_replicates_are_distinct_with_interior_minimizers():
    from neuralopt.task_families import family_objective as objective_of

    with jax.enable_x64(True):
        tasks = [make_task("rosenbrock-shifted", 0.0, 6, seed, objective_scale=5.0,
                           coordinate_scale=0.5) for seed in range(4)]
        minimizers = []
        for task in tasks:
            assert task.family == "rosenbrock-shifted" and task.known_optimum == 0.0
            objective = objective_of(task.family)
            x_star = task.theta["coordinate_scale"] * (1.0 + task.theta["shift"])
            assert bool(jnp.all((x_star > task.lo) & (x_star < task.hi)))
            np.testing.assert_allclose(objective(x_star, task.theta), 0.0, atol=1e-12)
            np.testing.assert_allclose(jax.grad(objective)(x_star, task.theta), 0.0, atol=1e-10)
            assert float(objective(task.x0, task.theta)) > 0
            minimizers.append(np.asarray(x_star))
        assert len({tuple(np.round(m, 6)) for m in minimizers}) == 4
    # The unshifted family still reproduces the v0.6 generator.
    assert make_task("rosenbrock", 0.0, 3, 1).family == "rosenbrock"
