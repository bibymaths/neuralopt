"""v0.8 problem families f(θ; c) and their fixed, hashed splits (plan sections 7-8)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt.families import FAMILIES, family_objective, make_split, split_digest
from neuralopt.lbfgs import lbfgs_solve, lbfgs_solve_batch


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


VARIANTS = ["sinusoid", "cascade-clean", "cascade-realistic"]


def _instance(name, split="validation", i=0):
    contexts, truths = make_split(name, split, i + 1)
    c = jax.tree_util.tree_map(lambda a: a[i], contexts)
    return FAMILIES[name], c, truths[i]


@pytest.mark.parametrize("name", VARIANTS)
def test_gradient_matches_central_differences(name):
    spec, c, truth = _instance(name)
    f = family_objective(name)
    key = jax.random.PRNGKey(3)
    x = truth + 0.05 * (spec.hi - spec.lo) * jax.random.uniform(key, truth.shape, minval=-1)
    x = jnp.clip(x, spec.lo, spec.hi)
    g = jax.grad(f)(x, c)
    h = 1e-6
    fd = jnp.asarray([(f(x + h * e, c) - f(x - h * e, c)) / (2 * h)
                      for e in jnp.eye(x.shape[0])])
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", VARIANTS)
def test_truth_fits_at_the_noise_level(name):
    contexts, truths = make_split(name, "validation", 50)
    f = family_objective(name)
    values = jax.vmap(f)(truths, contexts)
    # weighted least squares 0.5 * mean(((model - y) / sigma)^2): about 0.5 at the truth
    assert 0.35 <= float(jnp.mean(values)) <= 0.65


@pytest.mark.parametrize("name", ["cascade-clean", "cascade-realistic"])
def test_rk4_error_is_below_tolerance_against_a_16x_finer_reference(name):
    spec = FAMILIES[name]
    contexts, truths = make_split(name, "validation", 20)
    coarse = jax.vmap(lambda x, c: spec.observe(x, c, refine=1))(truths, contexts)
    fine = jax.vmap(lambda x, c: spec.observe(x, c, refine=16))(truths, contexts)
    scale = jnp.max(jnp.abs(fine))
    assert float(jnp.max(jnp.abs(coarse - fine)) / scale) <= spec.solver_tolerance


@pytest.mark.parametrize("name", VARIANTS)
def test_splits_are_deterministic_and_disjoint(name):
    a = split_digest(*make_split(name, "train", 30))
    assert a == split_digest(*make_split(name, "train", 30))
    others = {split_digest(*make_split(name, s, 30)) for s in FAMILIES[name].splits}
    assert len(others) == len(FAMILIES[name].splits)
    # instance i of a split does not depend on how many instances are requested
    small, _ = make_split(name, "test", 3)
    large, _ = make_split(name, "test", 10)
    np.testing.assert_array_equal(small["y"], large["y"][:3])


@pytest.mark.parametrize("name", VARIANTS)
def test_in_distribution_truths_lie_in_the_prior_and_ood_params_outside(name):
    spec = FAMILIES[name]
    _, truths = make_split(name, "test", 200)
    assert bool(jnp.all((truths >= spec.prior_lo) & (truths <= spec.prior_hi)))
    _, ood = make_split(name, "ood-params", 200)
    assert bool(jnp.all((ood >= spec.lo) & (ood <= spec.hi)))
    outside = (ood < spec.prior_lo) | (ood > spec.prior_hi)
    assert bool(jnp.all(jnp.any(outside, axis=1)))


def test_realistic_cascade_design_is_sparse_partial_and_dosed():
    spec = FAMILIES["cascade-realistic"]
    contexts, _ = make_split("cascade-realistic", "validation", 100)
    times_per_instance = jax.vmap(lambda c: jnp.unique(
        jnp.where(c["mask"], c["t"], -1.0), size=spec.max_observations, fill_value=-1.0)
    )(contexts)
    counts = jnp.sum(times_per_instance >= 0, axis=1) - 0  # distinct observed times
    assert int(counts.min()) >= 5 and int(counts.max()) <= 8
    p_obs = jnp.sum(contexts["mask"] & (contexts["species"] == 1), axis=1)
    i_obs = jnp.sum(contexts["mask"] & (contexts["species"] == 0), axis=1)
    assert bool(jnp.all(p_obs == counts)) and bool(jnp.all(i_obs < p_obs))
    doses = contexts["dose"]
    assert float(doses.min()) >= 0.5 and float(doses.max()) <= 5.0
    _, _ = make_split("cascade-realistic", "ood-dose", 5)
    assert float(make_split("cascade-realistic", "ood-dose", 50)[0]["dose"].min()) >= 6.0


def test_sinusoid_family_has_multiple_local_optima():
    """Known multimodality: different starts converge to different optima on some instances."""
    name = "sinusoid"
    spec = FAMILIES[name]
    f = family_objective(name)
    contexts, _ = make_split(name, "validation", 6)
    starts = spec.lo + (spec.hi - spec.lo) * jax.random.uniform(
        jax.random.PRNGKey(0), (16, spec.dimension))
    distinct = []
    for i in range(6):
        c = jax.tree_util.tree_map(lambda a, i=i: a[i], contexts)
        values = [lbfgs_solve(f, x0, (spec.lo, spec.hi), 400, theta=c)["fun"] for x0 in starts]
        distinct.append(len(np.unique(np.round(np.asarray(values), 4))))
    assert max(distinct) >= 3


@pytest.mark.parametrize("name", VARIANTS)
def test_batched_lbfgs_runs_on_a_split(name):
    spec = FAMILIES[name]
    contexts, _ = make_split(name, "validation", 4)
    x0 = jnp.tile(spec.default, (4, 1))
    lo, hi = jnp.tile(spec.lo, (4, 1)), jnp.tile(spec.hi, (4, 1))
    out = lbfgs_solve_batch(family_objective(name), contexts, x0, (lo, hi), 60)
    assert np.all(np.isfinite(out["fun"])) and np.all(out["used"] <= 60)
