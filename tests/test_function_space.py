import jax
import jax.numpy as jnp
import numpy as np

from neuralopt.function_space import (
    gaussian_reference_spectrum,
    periodic_reconstruction_objective,
    periodic_reconstruction_solution,
    sample_gaussian_field,
)


def test_periodic_reconstruction_refines_consistently():
    regularization = 0.2
    errors = []
    for size in (32, 64, 128, 256):
        grid = 2 * jnp.pi * jnp.arange(size) / size
        observations = jnp.sin(2 * grid) + 0.3 * jnp.cos(3 * grid)
        solution = periodic_reconstruction_solution(observations, regularization)
        expected = jnp.sin(2 * grid) / (1 + 4 * regularization)
        expected += 0.3 * jnp.cos(3 * grid) / (1 + 9 * regularization)
        errors.append(float(jnp.sqrt(jnp.mean((solution - expected) ** 2))))
        gradient = jax.grad(periodic_reconstruction_objective)(
            solution, observations, regularization
        )
        assert float(jnp.linalg.norm(gradient)) < 2e-5
    assert max(errors) < 2e-6


def test_gaussian_reference_is_reproducible_and_resolution_stable():
    key = jax.random.PRNGKey(3)
    first = sample_gaussian_field(key, 64)
    second = sample_gaussian_field(key, 64)
    np.testing.assert_array_equal(first, second)
    for size in (32, 64, 128, 256):
        spectrum = gaussian_reference_spectrum(size)
        assert np.all(np.diff(np.asarray(spectrum)) <= 0)
        assert np.isfinite(sample_gaussian_field(jax.random.PRNGKey(size), size)).all()
