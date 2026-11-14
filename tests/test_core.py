from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import ModelConfig, Problem, Variable, init_model, minimize, verify_certificate
from neuralopt.features import parameter_features
from neuralopt.model import apply_model, precondition


def test_problem_validation():
    with pytest.raises(ValueError):
        Problem(lambda x: jnp.sum(x), (1, -1), 2)
    with pytest.raises(ValueError):
        Problem(lambda x: jnp.sum(x), (-1, 1), 2, x0=jnp.zeros(3))


def test_model_shapes_and_positive_preconditioner():
    c = ModelConfig(width=16, heads=4, rank=2, layers=1)
    p = init_model(jax.random.PRNGKey(0), c)
    f = parameter_features(jnp.zeros(7), jnp.ones(7), -jnp.ones(7), jnp.ones(7), jnp.zeros(7))
    out = jax.jit(lambda z: apply_model(p, z, 1.0, c))(f)
    assert out["direction"].shape == (7,) and out["lowrank"].shape == (7, 2)
    assert np.all(np.asarray(out["diag"]) > 0)
    v = jnp.arange(7.0)
    pv = precondition(v, out["diag"], out["lowrank"])
    assert float(v @ pv) > 0


def test_deterministic_learned_minimize_and_bounds():
    problem = Problem(lambda x: jnp.sum((x - 0.3) ** 2), (-1.0, 1.0), 3, max_evaluations=12)
    a = minimize(problem, seed=4)
    b = minimize(problem, seed=4)
    np.testing.assert_allclose(a.x, b.x)
    assert a.fun == b.fun
    assert np.all(np.asarray(a.x) >= -1) and np.all(np.asarray(a.x) <= 1)
    assert len(a.trajectory) == 12 and np.all(np.diff(a.trajectory) <= 0)


def test_certified_quadratic_and_replay():
    x = Variable()
    expr = (x[0] - 0.25) ** 2 + (x[1] + 0.5) ** 2
    problem = Problem(
        lambda z: jnp.sum((z - jnp.array([0.25, -0.5])) ** 2),
        (-1.0, 1.0),
        2,
        certified_expression=expr,
        identifier="q",
    )
    result = minimize(problem, mode="certified", absolute_gap=2e-2, max_evaluations=2000)
    assert result.status == "certified_gap"
    assert result.lower_bound <= 0 <= result.upper_bound
    assert verify_certificate(result.certificate)


def test_certified_incumbent_preserves_binary64_precision():
    x = Variable()
    expr = (x[0] - 0.1) ** 2
    problem = Problem(
        lambda z: (z[0] - 0.1) ** 2,
        (0.0, 1.0),
        1,
        certified_expression=expr,
    )
    result = minimize(problem, mode="certified", absolute_gap=1e-8, max_evaluations=1000)
    assert result.x.dtype == np.float64
    point_upper = float(expr.interval(result.x, result.x).hi)
    assert point_upper <= result.upper_bound


def test_verifier_rejects_incomplete_domain_coverage():
    x = Variable()
    expr = x[0] ** 2
    problem = Problem(lambda z: z[0] ** 2, (-1.0, 1.0), 1, certified_expression=expr)
    result = minimize(problem, mode="certified", absolute_gap=0.1, max_evaluations=100)
    forged = replace(
        result.certificate,
        incumbent_x=[1.0],
        upper_bound=1.1,
        lower_bound=0.97,
        absolute_gap=0.2,
        active_regions=[{"lo": [0.99], "hi": [1.0], "lower_bound": 0.97}],
    )
    assert not verify_certificate(forged)


def test_cancellation_uses_point_interval_upper_bound():
    x = Variable()
    expr = (x[0] + 1e16) - 1e16
    problem = Problem(lambda z: z[0], (0.5, 1.5), 1, certified_expression=expr)
    result = minimize(problem, mode="certified", absolute_gap=1.0, max_evaluations=1)
    assert result.upper_bound >= 0.5
    assert float(expr.interval(result.x, result.x).hi) <= result.upper_bound


def test_arbitrary_jax_fails_closed():
    p = Problem(lambda x: jnp.sin(x[0]), (-1.0, 1.0), 1)
    with pytest.raises(ValueError, match="requires a neuralopt Expr"):
        minimize(p, mode="certified")
