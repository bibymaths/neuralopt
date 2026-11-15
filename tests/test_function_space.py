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


def test_periodic_reconstruction_penalizes_even_grid_nyquist_mode():
    size = 32
    regularization = 0.2
    observations = (-1.0) ** jnp.arange(size)
    solution = periodic_reconstruction_solution(observations, regularization)
    gradient = jax.grad(periodic_reconstruction_objective)(
        solution, observations, regularization
    )
    assert float(jnp.linalg.norm(gradient)) < 2e-5
    assert periodic_reconstruction_objective(
        solution, observations, regularization
    ) < periodic_reconstruction_objective(observations, observations, regularization)


def test_gaussian_reference_is_reproducible_and_resolution_stable():
    key = jax.random.PRNGKey(3)
    first = sample_gaussian_field(key, 64)
    second = sample_gaussian_field(key, 64)
    np.testing.assert_array_equal(first, second)
    for size in (32, 64, 128, 256):
        spectrum = gaussian_reference_spectrum(size)
        assert np.all(np.diff(np.asarray(spectrum)) <= 0)
        assert np.isfinite(sample_gaussian_field(jax.random.PRNGKey(size), size)).all()

    keys = jax.random.split(jax.random.PRNGKey(8), 2_000)
    for size in (32, 128):
        def spatial_mean(draw_key, sample_size=size):
            return sample_gaussian_field(draw_key, sample_size).mean()

        spatial_means = jax.vmap(spatial_mean)(keys)
        # alpha=1 and power=2 give unit variance to the constant Fourier mode.
        np.testing.assert_allclose(np.var(np.asarray(spatial_means)), 1.0, atol=0.1)
