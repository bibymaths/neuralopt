"""v0.8-B synthetic families (plan docs/plans/neuralopt-v08b-step-policy.md, section 2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt.families import SYNTHETIC, family_objective, make_split, split_digest
from neuralopt.lbfgs import lbfgs_solve


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


NAMES = ["rastrigin", "ackley", "rosenbrock-shifted"]


def _one(contexts, i):
    return jax.tree_util.tree_map(lambda a: a[i], contexts)


@pytest.mark.parametrize("name", NAMES)
def test_every_split_has_its_global_minimum_zero_at_the_shift(name):
    f = family_objective(name)
    for split in SYNTHETIC[name].splits:
        contexts, truths = make_split(name, split, 20)
        values = jax.vmap(f)(truths, contexts)
        assert float(jnp.max(jnp.abs(values))) <= 1e-12, split
        grads = jax.vmap(jax.grad(f))(truths, contexts)
        assert float(jnp.max(jnp.abs(grads))) <= 1e-9, split
        lo, hi = SYNTHETIC[name].bounds(split)
        assert bool(jnp.all((truths > lo) & (truths < hi))), split


@pytest.mark.parametrize("name", NAMES)
def test_objective_is_positive_away_from_the_optimum_and_gradient_matches_fd(name):
    contexts, truths = make_split(name, "validation", 3)
    f = family_objective(name)
    c, x = _one(contexts, 0), truths[0] + 0.37
    assert float(f(x, c)) > 0
    g = jax.grad(f)(x, c)
    h = 1e-6
    fd = jnp.asarray([(f(x + h * e, c) - f(x - h * e, c)) / (2 * h) for e in jnp.eye(x.shape[0])])
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_rotations_are_orthogonal_and_rosenbrock_is_unrotated(name):
    contexts, _ = make_split(name, "validation", 10)
    r = contexts["rotation"]
    eye = jnp.eye(r.shape[-1])
    np.testing.assert_allclose(jnp.einsum("nij,nkj->nik", r, r), jnp.broadcast_to(eye, r.shape),
                               atol=1e-12)
    if name == "rosenbrock-shifted":
        np.testing.assert_array_equal(r, jnp.broadcast_to(eye, r.shape))
    else:
        assert float(jnp.max(jnp.abs(r - eye))) > 0.1


@pytest.mark.parametrize("name", NAMES)
def test_shift_dimension_and_amplitude_ranges_per_split(name):
    spec = SYNTHETIC[name]
    _, ind = make_split(name, "test", 200)
    assert ind.shape[1] == spec.dimension == 4
    assert float(jnp.max(jnp.abs(ind))) <= spec.shift_half
    _, ood = make_split(name, "ood-shift", 200)
    lo_band, hi_band = spec.ood_shift
    assert float(jnp.min(jnp.abs(ood))) >= lo_band and float(jnp.max(jnp.abs(ood))) <= hi_band
    _, wide = make_split(name, "ood-dimension", 5)
    assert wide.shape[1] == 8
    contexts, _ = make_split(name, "validation", 200)
    if name == "rastrigin":
        a = contexts["amplitude"]
        assert 5.0 <= float(a.min()) and float(a.max()) <= 15.0
        oa, _ = make_split(name, "ood-amplitude", 50)
        assert 15.0 <= float(oa["amplitude"].min()) and float(oa["amplitude"].max()) <= 25.0
    else:
        assert "ood-amplitude" not in spec.splits


@pytest.mark.parametrize("name", NAMES)
def test_start_points_lie_in_the_box_and_are_instance_seeded(name):
    contexts, truths = make_split(name, "validation", 50)
    lo, hi = SYNTHETIC[name].bounds("validation")
    x0 = contexts["x0"]
    assert bool(jnp.all((x0 >= lo) & (x0 <= hi)))
    assert float(jnp.min(jnp.linalg.norm(x0 - truths, axis=1))) > 1e-3
    again, _ = make_split(name, "validation", 50)
    np.testing.assert_array_equal(again["x0"], x0)


@pytest.mark.parametrize("name", NAMES)
def test_synthetic_splits_are_deterministic_disjoint_and_prefix_stable(name):
    digests = {split_digest(*make_split(name, s, 20)) for s in SYNTHETIC[name].splits}
    assert len(digests) == len(SYNTHETIC[name].splits)
    assert split_digest(*make_split(name, "train", 20)) == split_digest(
        *make_split(name, "train", 20))
    small, _ = make_split(name, "test", 3)
    large, _ = make_split(name, "test", 9)
    np.testing.assert_array_equal(small["shift"], large["shift"][:3])


def test_rastrigin_is_multimodal_for_local_search():
    name = "rastrigin"
    contexts, _ = make_split(name, "validation", 1)
    c = _one(contexts, 0)
    lo, hi = SYNTHETIC[name].bounds("validation")
    starts = lo + (hi - lo) * jax.random.uniform(jax.random.PRNGKey(1), (16, 4))
    finals = [lbfgs_solve(family_objective(name), x0, (lo, hi), 300, theta=c)["fun"]
              for x0 in starts]
    assert len(np.unique(np.round(np.asarray(finals), 3))) >= 3
