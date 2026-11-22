"""Track B: Gram-space history model — B1 contracts and the B2 algebraic gate (float64)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import Problem, minimize
from neuralopt.features import gradient_to_normalized, normalize_coordinates
from neuralopt.gram import (
    GRAM_SCHEMA,
    GramConfig,
    RejectionInfo,
    apply_gram_model,
    apply_inverse_metric,
    dense_inverse_metric,
    gram_basis,
    gram_tokens,
    init_gram_model,
    load_gram_checkpoint,
    save_gram_checkpoint,
    tau_scale,
)
from neuralopt.trajectory import empty_history, push_history

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gram_reference import bfgs_coefficients, two_loop_direction  # noqa: E402

CONFIG = GramConfig(history_length=6, width=16, heads=4, layers=1)


def _spd(key, d, condition):
    q, _ = jnp.linalg.qr(jax.random.normal(key, (d, d)))
    return q @ jnp.diag(jnp.geomspace(1.0, condition, d)) @ q.T


def _history_from_points(points, objective, pairs):
    """History in u = x coordinates (box [-1, 1]); every point actually evaluated."""
    lo, hi = -jnp.ones(points.shape[1]), jnp.ones(points.shape[1])
    h = empty_history(pairs, points.shape[1], points.dtype)
    for x in points:
        h = push_history(h, normalize_coordinates(x, lo, hi),
                         gradient_to_normalized(jax.grad(objective)(x), lo, hi), objective(x))
    return h


# --- B2 (a): the model can express L-BFGS exactly -----------------------------


@pytest.mark.parametrize(("d", "pairs", "seed"), [(10, 4, 0), (10, 6, 1), (16, 8, 2), (5, 5, 3)])
def test_gram_metric_with_bfgs_coefficients_equals_lbfgs_two_loop(d, pairs, seed):
    with jax.enable_x64(True):
        hessian = _spd(jax.random.PRNGKey(seed), d, 100.0)

        def f(x):
            return 0.5 * x @ hessian @ x

        points = 0.3 * jax.random.normal(jax.random.PRNGKey(seed + 10), (pairs + 1, d))
        h = _history_from_points(points, f, pairs)
        basis = gram_basis(h)
        assert bool(jnp.all(basis.mask)) and bool(jnp.all(basis.curvature > 0))
        S = basis.S_hat * basis.s_norm[None, :]
        Y = basis.Y_hat * basis.y_norm[None, :]
        g = jax.grad(f)(points[-1])
        tau = tau_scale(basis, g, CONFIG.fallback_length)
        gamma = (S[:, -1] @ Y[:, -1]) / (Y[:, -1] @ Y[:, -1])
        A_hat, C_hat, delta = bfgs_coefficients(S, Y, tau, gamma)
        # The BFGS values must lie inside the learned model's output ranges to count as
        # expressible (bound * tanh(raw / bound) reaches every value strictly inside).
        assert float(jnp.abs(A_hat).max()) < CONFIG.coefficient_bound
        assert float(jnp.abs(C_hat).max()) < CONFIG.coefficient_bound
        assert abs(float(delta)) < CONFIG.delta_bound
        ours = apply_inverse_metric(g, basis, A_hat, C_hat, delta, tau, eps_rel=0.0)
        reference = two_loop_direction(g, S, Y, gamma)
        np.testing.assert_allclose(ours, reference, rtol=1e-10, atol=1e-10 * float(
            jnp.linalg.norm(reference)))
        # Inverse-secant convention: the newest pair satisfies P y = s exactly (BFGS).
        newest = apply_inverse_metric(Y[:, -1], basis, A_hat, C_hat, delta, tau, eps_rel=0.0)
        np.testing.assert_allclose(newest, S[:, -1], rtol=1e-9, atol=1e-12)


# --- B2 (b): P is symmetric positive definite -----------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_gram_metric_is_spd_for_random_coefficients(seed):
    with jax.enable_x64(True):
        d, pairs = 9, 6
        keys = jax.random.split(jax.random.PRNGKey(seed), 5)
        points = jax.random.normal(keys[0], (pairs + 1, d)) * 0.4
        hessian = _spd(keys[1], d, 30.0)
        h = _history_from_points(points, lambda x: 0.5 * x @ hessian @ x, pairs)
        basis = gram_basis(h)
        A = 5.0 * jax.random.normal(keys[2], (pairs, pairs))
        C = 5.0 * jax.random.normal(keys[3], (pairs, pairs))
        delta = jax.random.normal(keys[4], ())
        tau = jnp.asarray(0.7)
        P = dense_inverse_metric(basis, A, C, delta, tau, eps_rel=1e-6)
        np.testing.assert_allclose(P, P.T, atol=1e-10 * float(jnp.abs(P).max()))
        eigenvalues = jnp.linalg.eigvalsh(P)
        assert float(eigenvalues.min()) >= 1e-6 * 0.7 * (1 - 1e-6)
        # Matrix-free application equals the dense matrix.
        v = jax.random.normal(keys[0], (d,))
        np.testing.assert_allclose(apply_inverse_metric(v, basis, A, C, delta, tau, 1e-6), P @ v,
                                   rtol=1e-10, atol=1e-10)


def test_masked_pairs_contribute_nothing():
    with jax.enable_x64(True):
        d, pairs = 6, 4
        hessian = _spd(jax.random.PRNGKey(0), d, 10.0)
        f = lambda x: 0.5 * x @ hessian @ x  # noqa: E731
        points = 0.3 * jax.random.normal(jax.random.PRNGKey(1), (3, d))  # only 2 valid pairs
        h = _history_from_points(points, f, pairs)
        basis = gram_basis(h)
        assert basis.mask.tolist() == [False, False, True, True]
        A = jax.random.normal(jax.random.PRNGKey(2), (pairs, pairs))
        C = jax.random.normal(jax.random.PRNGKey(3), (pairs, pairs))
        P = dense_inverse_metric(basis, A, C, 0.3, 1.0, 1e-8)
        A2 = A.at[:2, :].set(99.0).at[:, :2].set(-99.0)
        C2 = C.at[:2, :].set(42.0).at[:, :2].set(7.0)
        np.testing.assert_allclose(dense_inverse_metric(basis, A2, C2, 0.3, 1.0, 1e-8), P,
                                   atol=1e-12)


# --- B1 contracts: invariant tokens, model, schema ------------------------------


def test_tokens_are_rotation_and_objective_scale_invariant():
    with jax.enable_x64(True):
        d, pairs = 7, CONFIG.history_length
        hessian = _spd(jax.random.PRNGKey(4), d, 50.0)
        q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(5), (d, d)))
        points = 0.2 * jax.random.normal(jax.random.PRNGKey(6), (pairs + 1, d))

        def f(x):
            return 0.5 * x @ hessian @ x

        def f_rot(x):
            return f(q.T @ x)

        rejection = RejectionInfo(jnp.asarray(2), jnp.asarray(0.05), jnp.asarray(1.7),
                                  jnp.asarray(True))
        base = gram_tokens(_history_from_points(points, f, pairs), jax.grad(f)(points[-1]),
                           rejection, 0.4, CONFIG)
        rotated = gram_tokens(_history_from_points(points @ q.T, f_rot, pairs),
                              jax.grad(f_rot)(q @ points[-1]), rejection, 0.4, CONFIG)
        scaled = gram_tokens(_history_from_points(points, lambda x: 9.0 * f(x), pairs),
                             9.0 * jax.grad(f)(points[-1]), rejection, 0.4, CONFIG)
        assert base.shape == (pairs + 1, CONFIG.token_features)
        np.testing.assert_allclose(rotated, base, atol=1e-9)
        np.testing.assert_allclose(scaled, base, atol=1e-9)


def test_model_outputs_shapes_bounds_and_scalar_only_mode():
    params = init_gram_model(jax.random.PRNGKey(0), CONFIG)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (CONFIG.history_length + 1,
                                                       CONFIG.token_features))
    mask = jnp.array([False, True, True, True, True, True])
    out = apply_gram_model(params, tokens, mask, CONFIG)
    assert out["A"].shape == out["C"].shape == (6, 6)
    assert bool(jnp.all(out["A"][0] == 0)) and bool(jnp.all(out["C"][:, 0] == 0))
    assert 1 / CONFIG.sigma_bound <= float(out["sigma"]) <= CONFIG.sigma_bound
    assert abs(float(out["delta"])) <= CONFIG.delta_bound
    scalar = GramConfig(**{**CONFIG.__dict__, "scalar_only": True})
    out_scalar = apply_gram_model(params, tokens, mask, scalar)
    assert bool(jnp.all(out_scalar["A"] == 0)) and bool(jnp.all(out_scalar["C"] == 0))


def test_gram_checkpoint_roundtrip_and_fail_closed(tmp_path):
    from neuralopt.training import load_checkpoint

    params = init_gram_model(jax.random.PRNGKey(0), CONFIG)
    path = tmp_path / "gram.npz"
    save_gram_checkpoint(path, params, CONFIG, {"seed": 0})
    loaded, config = load_gram_checkpoint(path)
    assert config == CONFIG
    for k in params:
        np.testing.assert_array_equal(loaded[k], params[k])
    with np.load(path) as data:
        assert str(data["schema"][0]) == GRAM_SCHEMA
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path)  # legacy/history loader refuses a Gram file


# --- B2 (c): equivariance of the whole trajectory -------------------------------


def _quadratic(hessian, center):
    def f(x):
        return 0.5 * (x - center) @ hessian @ (x - center)

    return f


def _solve(objective, x0, bound, params, config, budget=60):
    d = x0.shape[0]
    problem = Problem(objective, (-bound * np.ones(d), bound * np.ones(d)), d,
                      x0=np.asarray(x0), dtype=jnp.float64)
    return minimize(problem, max_evaluations=budget, params=params, config=config,
                    model_kind="gram", candidate_method="preconditioned",
                    preconditioned_engine="compiled", record_diagnostics=True)


def test_trajectory_is_rotation_equivariant_and_objective_scale_invariant():
    with jax.enable_x64(True):
        d, bound = 8, 1000.0  # huge box: clipping never active for this interior optimum
        hessian = _spd(jax.random.PRNGKey(7), d, 100.0)
        center = 0.5 * jax.random.normal(jax.random.PRNGKey(8), (d,))
        q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(9), (d, d)))
        x0 = jnp.ones(d) * 2.0
        params = init_gram_model(jax.random.PRNGKey(10), CONFIG)
        f = _quadratic(hessian, center)
        base = _solve(f, x0, bound, params, CONFIG)
        rotated = _solve(_quadratic(q @ hessian @ q.T, q @ center), q @ x0, bound, params, CONFIG)
        scaled = _solve(lambda x: 250.0 * f(x), x0, bound, params, CONFIG)
        candidates = np.array([r["candidate"] for r in base.diagnostics["records"]])
        assert np.abs(candidates).max() < 0.5 * bound  # clip inactive
        assert base.diagnostics["accepted_steps"] >= 5
        rot_candidates = np.array([r["candidate"] for r in rotated.diagnostics["records"]])
        np.testing.assert_allclose(rot_candidates, candidates @ np.asarray(q).T, rtol=1e-7,
                                   atol=1e-8)
        np.testing.assert_allclose(rotated.trajectory, base.trajectory, rtol=1e-7)
        np.testing.assert_allclose(np.array(scaled.trajectory) / 250.0, base.trajectory,
                                   rtol=1e-7)
        assert [r["accepted"] for r in scaled.diagnostics["records"]] == [
            r["accepted"] for r in base.diagnostics["records"]]


def test_gram_records_report_tau_and_sigma_separately():
    with jax.enable_x64(True):
        d = 6
        hessian = _spd(jax.random.PRNGKey(11), d, 50.0)
        f = _quadratic(hessian, jnp.zeros(d))
        params = init_gram_model(jax.random.PRNGKey(12), CONFIG)
        base = _solve(f, jnp.ones(d), 10.0, params, CONFIG, budget=30)
        tiny = _solve(lambda x: 1e-6 * f(x), jnp.ones(d), 10.0, params, CONFIG, budget=30)
    rb, rt = base.diagnostics["records"], tiny.diagnostics["records"]
    assert all(r["tau"] > 0 and 1 / CONFIG.sigma_bound <= r["sigma"] <= CONFIG.sigma_bound
               for r in rb)
    # tau carries the inverse objective units; sigma is dimensionless.
    np.testing.assert_allclose([r["tau"] * 1e-6 for r in rt], [r["tau"] for r in rb], rtol=1e-6)
    np.testing.assert_allclose([r["sigma"] for r in rt], [r["sigma"] for r in rb], rtol=1e-9)
