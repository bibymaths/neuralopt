import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import Problem
from neuralopt.features import gradient_to_normalized
from neuralopt.model import ModelConfig
from neuralopt.tasks import task_split
from neuralopt.training import (
    StochasticTrainingConfig,
    UnrollConfig,
    _quadratic_euler_weak_residual,
    load_checkpoint,
    save_checkpoint,
    stochastic_geometry_loss,
    train,
    unrolled_regret_loss,
)

pytest.importorskip("optax")


def test_split_checkpoint_and_training(tmp_path):
    records = task_split(3, 20, 3)
    assert [sum(r.split == s for r in records) for s in ("train", "validation", "test")] == [
        14,
        3,
        3,
    ]
    params, loss = train(
        [r.problem for r in records if r.split == "train"],
        seed=3,
        steps=12,
        config=ModelConfig(width=16, heads=4, rank=1, layers=1),
    )
    assert np.isfinite(loss).all()
    path = tmp_path / "model.npz"
    save_checkpoint(path, params, ModelConfig(width=16, heads=4, rank=1, layers=1), {"seed": 3})
    loaded, config = load_checkpoint(path)
    assert config.width == 16
    for a, b in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(loaded), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_unrolled_loss_has_finite_nonzero_meta_gradient():
    problem = task_split(5, 5, 2)[0].problem
    config = ModelConfig(width=8, heads=2, rank=1, layers=1)
    from neuralopt.model import init_model

    params = init_model(jax.random.PRNGKey(0), config)
    loss, grad = jax.value_and_grad(unrolled_regret_loss)(
        params,
        problem,
        jax.random.PRNGKey(1),
        config,
        UnrollConfig(steps=5, horizons=(1, 3, 5)),
    )
    flat = np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(grad)])
    assert np.isfinite(loss) and np.isfinite(flat).all()
    assert np.linalg.norm(flat) > 0


def test_mlp_baseline_can_train_and_run():
    problem = task_split(8, 5, 2)[0].problem
    config = ModelConfig(width=8, heads=2, rank=1, layers=1)
    params, losses = train(
        [problem],
        steps=2,
        config=config,
        objective="regret",
        unroll=UnrollConfig(steps=3, horizons=(1, 3)),
        model_kind="mlp",
    )
    from neuralopt import minimize

    result = minimize(problem, max_evaluations=4, params=params, config=config, model_kind="mlp")
    assert np.isfinite(losses).all() and np.isfinite(result.fun)


def test_regret_requires_known_optimum():
    problem = Problem(lambda x: jnp.sum(x**2) - 100.0, (-1.0, 1.0), 2)
    config = ModelConfig(width=8, heads=2, rank=1, layers=1)
    from neuralopt.model import init_model

    params = init_model(jax.random.PRNGKey(0), config)
    with pytest.raises(ValueError, match="known_optimum"):
        unrolled_regret_loss(
            params,
            problem,
            jax.random.PRNGKey(1),
            config,
            UnrollConfig(steps=2, horizons=(1, 2)),
        )


def test_stationary_unroll_has_finite_meta_gradient(monkeypatch):
    problem = Problem(
        lambda x: jnp.sum(x * x),
        (-1.0, 1.0),
        2,
        metadata={"known_optimum": 0.0},
    )
    config = ModelConfig(width=8, heads=2, rank=1, layers=1)
    from neuralopt.model import init_model

    params = init_model(jax.random.PRNGKey(0), config)

    # Force the unroll's sampled start to the exact stationary point.
    def zeros(*args, **kwargs):
        return jnp.zeros(kwargs.get("shape", args[1]))

    monkeypatch.setattr(jax.random, "uniform", zeros)
    loss, grad = jax.value_and_grad(unrolled_regret_loss)(
        params,
        problem,
        jax.random.PRNGKey(1),
        config,
        UnrollConfig(steps=3, horizons=(1, 3)),
    )
    leaves = jax.tree_util.tree_leaves(grad)
    assert np.isfinite(loss) and all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)


def test_direction_alignment_uses_normalized_coordinates():
    lo = jnp.array([0.0, 0.0])
    hi = jnp.array([100.0, 1.0])
    gradient = jnp.array([1.0, 1.0])
    normalized_direction = jnp.array([0.1, -0.995])
    # Mixing coordinate systems incorrectly labels this direction as descent.
    assert float(normalized_direction @ gradient) < 0
    normalized_gradient = gradient_to_normalized(gradient, lo, hi)
    # The actual physical update scales by box width and is ascent.
    assert float(normalized_direction @ normalized_gradient) > 0


@pytest.mark.parametrize("integrator", ["overdamped", "underdamped"])
def test_stochastic_geometry_loss_propagates_finite_geometry_gradients(integrator):
    problem = task_split(11, 4, 2)[0].problem
    config = ModelConfig(width=8, heads=2, rank=1, layers=1)
    from neuralopt.model import init_model

    params = init_model(jax.random.PRNGKey(0), config)
    stochastic = StochasticTrainingConfig(
        integrator=integrator,
        episodes=3,
        integration_steps=1,
        horizons=(1, 3),
        fp_weight=0.1 if integrator == "overdamped" else 0.0,
    )
    def loss_fn(p):
        return stochastic_geometry_loss(p, problem, jax.random.PRNGKey(2), config, stochastic)

    first, gradients = jax.value_and_grad(loss_fn)(params)
    second = loss_fn(params)
    np.testing.assert_allclose(first, second, rtol=1e-6)
    geometry = np.concatenate(
        [np.asarray(gradients[name]).ravel() for name in ("diag", "lowrank")]
    )
    assert np.isfinite(first) and np.isfinite(geometry).all()
    assert np.linalg.norm(geometry) > 0


def test_stochastic_fp_configuration_and_training():
    with pytest.raises(ValueError, match="overdamped"):
        StochasticTrainingConfig(integrator="underdamped", fp_weight=0.1)
    problem = task_split(12, 2, 2)[0].problem
    config = ModelConfig(width=8, heads=2, rank=1, layers=1)
    params, losses = train(
        [problem],
        seed=2,
        steps=2,
        learning_rate=1e-3,
        config=config,
        objective="stochastic",
        stochastic=StochasticTrainingConfig(
            episodes=3, integration_steps=1, horizons=(1, 3), fp_weight=0.05
        ),
    )
    assert np.isfinite(losses).all()
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(params))


def test_quadratic_weak_residual_integrates_noise_variance_exactly():
    position = jnp.array([-0.7, 0.4, 1.2])
    for covariance_trace in (0.1, 10.0, 1_000.0):
        residual = _quadratic_euler_weak_residual(
            position, jnp.zeros_like(position), covariance_trace, 0.01
        )
        np.testing.assert_allclose(residual, 0.0, atol=2e-5)
