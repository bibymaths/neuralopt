import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import Problem, init_model, minimize
from neuralopt.langevin import (
    OverdampedConfig,
    UnderdampedConfig,
    diffrax_overdamped_reference,
    overdamped_transition,
    sample_mobility_noise,
    underdamped_transition,
)


def test_mobility_noise_covariance_and_reproducibility():
    diag = jnp.array([0.8, 1.4])
    lowrank = jnp.array([[0.3], [-0.4]])
    expected = np.diag(np.asarray(diag)) + np.asarray(lowrank) @ np.asarray(lowrank).T
    keys = jax.random.split(jax.random.PRNGKey(0), 20_000)
    samples = jax.vmap(lambda key: sample_mobility_noise(key, diag, lowrank))(keys)
    np.testing.assert_allclose(np.cov(np.asarray(samples), rowvar=False), expected, atol=0.035)
    key = jax.random.PRNGKey(7)
    np.testing.assert_array_equal(
        sample_mobility_noise(key, diag, lowrank),
        sample_mobility_noise(key, diag, lowrank),
    )


def test_overdamped_gaussian_temperature_scaling():
    def objective(x):
        return 0.5 * jnp.sum(x * x)

    lo, hi = -8.0 * jnp.ones(1), 8.0 * jnp.ones(1)

    def variance(temperature):
        state = jnp.zeros(1)
        z = None
        key = jax.random.PRNGKey(int(100 * temperature))
        draws = []
        config = OverdampedConfig(step_size=0.015, temperature=temperature, steps=4)
        for iteration in range(1_200):
            key, transition_key = jax.random.split(key)
            transition = overdamped_transition(
                transition_key,
                state,
                objective,
                lo,
                hi,
                jnp.ones(1),
                jnp.zeros((1, 0)),
                config,
                unconstrained_state=z,
            )
            assert transition.finite
            state, z = transition.state, transition.unconstrained_state
            if iteration >= 200:
                draws.append(float(state[0]))
        return np.var(draws)

    low_temperature = variance(0.5)
    high_temperature = variance(1.0)
    assert 0.3 < low_temperature < 0.8
    assert 0.65 < high_temperature < 1.4
    assert high_temperature > low_temperature


def test_underdamped_baoab_kinetic_temperature_and_bounds():
    def objective(x):
        return 0.5 * jnp.sum(x * x)

    lo, hi = -8.0 * jnp.ones(1), 8.0 * jnp.ones(1)
    state, z, momentum = jnp.zeros(1), None, None
    key = jax.random.PRNGKey(5)
    positions, momenta = [], []
    config = UnderdampedConfig(step_size=0.03, temperature=0.7, friction=1.2, steps=3)
    for iteration in range(1_500):
        key, transition_key = jax.random.split(key)
        transition = underdamped_transition(
            transition_key,
            state,
            objective,
            lo,
            hi,
            jnp.ones(1),
            jnp.zeros((1, 0)),
            config,
            unconstrained_state=z,
            momentum=momentum,
        )
        assert transition.finite
        state, z, momentum = (
            transition.state,
            transition.unconstrained_state,
            transition.momentum,
        )
        if iteration >= 300:
            positions.append(float(state[0]))
            momenta.append(float(momentum[0]))
    assert np.all(np.abs(positions) < 8.0)
    np.testing.assert_allclose(np.var(positions), 0.7, atol=0.2)
    np.testing.assert_allclose(np.var(momenta), 0.7, atol=0.2)


def test_langevin_invalid_configuration_and_nonfinite_status():
    with pytest.raises(ValueError, match="positive"):
        OverdampedConfig(temperature=0)

    def invalid_objective(x):
        return jnp.sum(x) * jnp.nan

    transition = overdamped_transition(
        jax.random.PRNGKey(0),
        jnp.zeros(1),
        invalid_objective,
        -jnp.ones(1),
        jnp.ones(1),
        jnp.ones(1),
        jnp.zeros((1, 0)),
    )
    assert not transition.finite


@pytest.mark.parametrize("method", ["overdamped", "underdamped"])
def test_langevin_rejects_nonfinite_unconstrained_position(method):
    def steep_objective(x):
        return 1e20 * jnp.sum((x - 0.5) ** 2)

    common = (
        jax.random.PRNGKey(0),
        jnp.zeros(1),
        steep_objective,
        -jnp.ones(1),
        jnp.ones(1),
        jnp.ones(1),
        jnp.zeros((1, 0)),
    )
    if method == "overdamped":
        transition = overdamped_transition(
            *common, config=OverdampedConfig(step_size=1e20)
        )
    else:
        transition = underdamped_transition(
            *common, config=UnderdampedConfig(step_size=1e20)
        )
    assert not transition.finite
    assert not np.isfinite(transition.unconstrained_state).all()


@pytest.mark.parametrize("method", ["langevin", "underdamped"])
def test_stochastic_optimizer_preserves_boundary_incumbent(method):
    lo, hi = -jnp.ones(2), 2.0 * jnp.ones(2)
    problem = Problem(lambda x: jnp.sum((x - lo) ** 2), (lo, hi), 2, x0=lo)
    result = minimize(
        problem,
        max_evaluations=40,
        seed=3,
        params=init_model(jax.random.PRNGKey(0)),
        candidate_method=method,
        overdamped_config=OverdampedConfig(step_size=0.01, steps=2),
        underdamped_config=UnderdampedConfig(step_size=0.01, steps=2),
        record_diagnostics=True,
    )
    np.testing.assert_array_equal(result.x, lo)
    assert result.fun == 0.0
    assert np.all(np.diff(result.trajectory) <= 0)
    assert result.diagnostics["stochastic_integration_steps"] > 0
    assert result.diagnostics["gradient_evaluations"] > 0


def test_stochastic_optimizer_accepts_valid_fixed_diagonal_geometry():
    problem = Problem(lambda x: jnp.sum(x**2), (-2.0, 2.0), 2)
    params = init_model(jax.random.PRNGKey(0))
    result = minimize(
        problem,
        max_evaluations=12,
        params=params,
        candidate_method="langevin",
        hamiltonian_metric="fixed",
        fixed_metric_diag=jnp.array([0.5, 2.0]),
    )
    assert np.isfinite(result.fun)
    with pytest.raises(ValueError, match="fixed_metric_diag"):
        minimize(
            problem,
            max_evaluations=12,
            params=params,
            candidate_method="langevin",
            hamiltonian_metric="fixed",
        )


def test_optional_diffrax_reference_executes_reproducibly():
    pytest.importorskip("diffrax")

    def objective(x):
        return 0.5 * jnp.sum(x * x)

    args = (jax.random.PRNGKey(2), jnp.array([0.3, -0.2]), objective, 0.01, 5, 0.7)
    first = diffrax_overdamped_reference(*args)
    second = diffrax_overdamped_reference(*args)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
