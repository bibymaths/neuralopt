import jax
import jax.numpy as jnp
import numpy as np

from neuralopt.fokker_planck import gaussian_free_energy, generator_value, weak_form_residual
from neuralopt.stochastic_geometry import ito_gibbs_drift, mobility_divergence


def test_exact_mobility_divergence_and_gibbs_drift():
    def mobility(x):
        return jnp.diag(1.0 + x * x)

    x = jnp.array([0.3, -0.4])
    np.testing.assert_allclose(mobility_divergence(mobility, x), 2.0 * x)

    def objective(y):
        return 0.5 * y @ y

    expected = -(1.0 + x * x) * x + 0.7 * (2.0 * x)
    np.testing.assert_allclose(ito_gibbs_drift(objective, mobility, x, 0.7), expected)


def test_generator_and_weak_form_ou_identity():
    def test_function(x):
        return jnp.sum(x * x)

    x = jnp.array([0.2, -0.5])
    drift = -x
    covariance = 2.0 * jnp.eye(2)
    expected = -2.0 * jnp.sum(x * x) + 4.0
    np.testing.assert_allclose(generator_value(test_function, x, drift, covariance), expected)

    key_x, key_noise = jax.random.split(jax.random.PRNGKey(2))
    base_samples = jax.random.normal(key_x, (20_000, 2))
    samples = jnp.concatenate((base_samples, base_samples))
    base_noise = jax.random.normal(key_noise, base_samples.shape)
    noise = jnp.concatenate((base_noise, -base_noise))
    dt = 2e-2
    next_samples = samples - dt * samples + jnp.sqrt(2.0 * dt) * noise
    residual = weak_form_residual(
        samples,
        next_samples,
        dt,
        test_function,
        lambda point: -point,
        lambda point: 2.0 * jnp.eye(point.shape[0]),
    )
    assert abs(float(residual)) < 0.2


def test_gaussian_free_energy_minimum_for_quadratic_gibbs_state():
    quadratic = jnp.diag(jnp.array([1.0, 4.0]))
    temperature = 0.5
    equilibrium = temperature * jnp.linalg.inv(quadratic)
    optimum = gaussian_free_energy(jnp.zeros(2), equilibrium, quadratic, temperature)
    displaced = gaussian_free_energy(
        jnp.array([0.4, -0.2]), 1.8 * equilibrium, quadratic, temperature
    )
    assert displaced > optimum
