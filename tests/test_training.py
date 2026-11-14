import jax
import numpy as np
import pytest

from neuralopt.model import ModelConfig
from neuralopt.tasks import task_split
from neuralopt.training import (
    UnrollConfig,
    load_checkpoint,
    save_checkpoint,
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
