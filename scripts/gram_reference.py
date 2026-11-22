"""Harness-only BFGS references for the Gram-space model gates (B2, B3).

Never imported by ``src/neuralopt``: NeuralOpt inference must not contain a BFGS mode.

Derivation (Byrd, Nocedal & Schnabel 1994, compact representation), with ``H0 = γ I``::

    H = γ I + S R⁻ᵀ (D + γ YᵀY) R⁻¹ Sᵀ − γ S R⁻ᵀ Yᵀ − γ Y R⁻¹ Sᵀ,
    R_ij = s_iᵀ y_j (i <= j, chronological), D = diag(s_iᵀ y_i).

The Gram model's inverse metric (ε = 0)::

    P = τ [ e^δ (I − S A Yᵀ)(I − Y Aᵀ Sᵀ) + S C Cᵀ Sᵀ ]

equals H for τ e^δ = γ, A = R⁻ᵀ and τ C Cᵀ = R⁻ᵀ D R⁻¹, i.e. C = R⁻ᵀ D^{1/2} / √τ
(requires every s_iᵀ y_i > 0). With column-normalized Ŝ = S D_s⁻¹, Ŷ = Y D_y⁻¹ the
coefficients become Â = D_s A D_y and Ĉ = D_s C.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def two_loop_direction(g: Any, S: Any, Y: Any, gamma: Any) -> Any:
    """L-BFGS two-loop ``H g`` with ``H0 = γ I``; columns of S, Y oldest -> newest."""
    k = S.shape[1]
    rho = 1.0 / jnp.sum(S * Y, axis=0)
    q = g
    alpha = [None] * k
    for i in reversed(range(k)):
        alpha[i] = rho[i] * (S[:, i] @ q)
        q = q - alpha[i] * Y[:, i]
    r = gamma * q
    for i in range(k):
        beta = rho[i] * (Y[:, i] @ r)
        r = r + S[:, i] * (alpha[i] - beta)
    return r


def bfgs_coefficients(S: Any, Y: Any, tau: Any, gamma: Any) -> tuple[Any, Any, Any]:
    """Normalized-column ``(Â, Ĉ, δ)`` reproducing L-BFGS with ``H0 = γ I``.

    ``S``, ``Y``: (d, K) valid pairs, chronological, all with ``s_iᵀ y_i > 0``.
    """
    sty = S.T @ Y
    R = jnp.triu(sty)
    D = jnp.diag(sty)
    if bool(jnp.any(D <= 0)):
        raise ValueError("BFGS oracle needs s_i^T y_i > 0 for every pair")
    R_inv_T = jnp.linalg.inv(R).T
    A = R_inv_T
    C = R_inv_T * jnp.sqrt(D)[None, :] / jnp.sqrt(tau)
    s_norm = jnp.linalg.norm(S, axis=0)
    y_norm = jnp.linalg.norm(Y, axis=0)
    A_hat = s_norm[:, None] * A * y_norm[None, :]
    C_hat = s_norm[:, None] * C
    delta = jnp.log(gamma / tau)
    return A_hat, C_hat, delta
