"""Periodic function-space reference geometry for refinement experiments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def fourier_wavenumbers(size: int, length: float = 2 * jnp.pi) -> Any:
    if size < 2 or length <= 0:
        raise ValueError("size must be at least two and length must be positive")
    spacing = length / size
    return 2 * jnp.pi * jnp.fft.rfftfreq(size, d=spacing)


def gaussian_reference_spectrum(
    size: int, alpha: float = 1.0, power: float = 2.0, length: float = 2 * jnp.pi
) -> Any:
    """Trace-class-like discretization ``(alpha + |k|^2)^-power``."""
    if alpha <= 0 or power <= 0.5:
        raise ValueError("alpha must be positive and power must exceed one half")
    wave = fourier_wavenumbers(size, length)
    return (alpha + wave * wave) ** (-power)


def sample_gaussian_field(
    key: Any,
    size: int,
    alpha: float = 1.0,
    power: float = 2.0,
) -> Any:
    """Sample a real periodic field with resolution-stable spectral covariance."""
    white = jax.random.normal(key, (size,))
    coefficients = jnp.fft.rfft(white)
    spectrum = gaussian_reference_spectrum(size, alpha, power)
    # With JAX's default FFT normalization, the constant coefficient of grid
    # white noise has spatial-mean variance 1 / size. The square-root factor
    # converts to the mesh-normalized Fourier basis, keeping every retained
    # mode's continuum covariance stable under refinement.
    return jnp.fft.irfft(coefficients * jnp.sqrt(size * spectrum), n=size)


def periodic_reconstruction_objective(
    field: Any, observations: Any, regularization: float, length: float = 2 * jnp.pi
) -> Any:
    """Mesh-normalized L2 fit plus periodic H1 seminorm."""
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")
    wave = fourier_wavenumbers(field.shape[0], length)
    size = field.shape[0]
    coefficients = jnp.fft.rfft(field)
    # Parseval weights for an rFFT: interior positive frequencies represent
    # conjugate pairs, whereas zero and the even-grid Nyquist mode do not.
    # Computing the seminorm spectrally also gives the real Nyquist coefficient
    # the same k^2 penalty used by periodic_reconstruction_solution.
    weights = jnp.ones_like(wave).at[1:].set(2.0)
    if size % 2 == 0:
        weights = weights.at[-1].set(1.0)
    derivative_energy = jnp.sum(weights * wave * wave * jnp.abs(coefficients) ** 2) / size**2
    return 0.5 * jnp.mean((field - observations) ** 2) + 0.5 * regularization * derivative_energy


def periodic_reconstruction_solution(
    observations: Any, regularization: float, length: float = 2 * jnp.pi
) -> Any:
    """Exact discrete minimizer for :func:`periodic_reconstruction_objective`."""
    wave = fourier_wavenumbers(observations.shape[0], length)
    coefficients = jnp.fft.rfft(observations)
    return jnp.fft.irfft(
        coefficients / (1.0 + regularization * wave * wave), n=observations.shape[0]
    )
