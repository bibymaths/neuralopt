import jax
import numpy as np
import pytest

from neuralopt.model import ModelConfig
from neuralopt.tasks import task_split
from neuralopt.training import load_checkpoint, save_checkpoint, train

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
