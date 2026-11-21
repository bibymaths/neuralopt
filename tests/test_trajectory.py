"""v0.6 trajectory history: secant information, masking, invariances, JAX transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt.features import gradient_to_normalized, normalize_coordinates
from neuralopt.trajectory import (
    HISTORY_FEATURE_NAMES,
    HISTORY_FEATURES,
    empty_history,
    history_features,
    push_history,
    record_rejection,
    secant_pairs,
)

IDX = {name: i for i, name in enumerate(HISTORY_FEATURE_NAMES)}
GLOBAL = [n for n in HISTORY_FEATURE_NAMES if n not in ("diag_curvature", "last_s", "last_y")]


def _spd(key, d, condition=100.0):
    q, _ = jnp.linalg.qr(jax.random.normal(key, (d, d)))
    return q @ jnp.diag(jnp.geomspace(1.0, condition, d)) @ q.T


def _fill(objective, xs, lo, hi, pairs):
    """Push actually evaluated (u, g_u, f) records for physical points ``xs``."""
    h = empty_history(pairs, xs.shape[1])
    grad = jax.grad(objective)
    for x in xs:
        h = push_history(
            h,
            normalize_coordinates(x, lo, hi),
            gradient_to_normalized(grad(x), lo, hi),
            objective(x),
        )
    return h


def _current_gu(objective, x, lo, hi):
    return gradient_to_normalized(jax.grad(objective)(x), lo, hi)


def test_feature_width_is_twelve():
    assert HISTORY_FEATURES == len(HISTORY_FEATURE_NAMES) == 12


def test_secant_is_exact_hessian_action_in_normalized_frame():
    d = 4
    lo = jnp.array([-1.0, -3.0, 0.0, -10.0])
    hi = jnp.array([2.0, 1.0, 5.0, 10.0])
    hx = _spd(jax.random.PRNGKey(0), d)
    center = jnp.array([0.5, -1.0, 2.0, 3.0])

    def f(x):
        return 0.5 * (x - center) @ hx @ (x - center)

    xs = jax.random.uniform(jax.random.PRNGKey(1), (5, d), minval=lo, maxval=hi)
    s, y, mask = secant_pairs(_fill(f, xs, lo, hi, 4))
    jac = jnp.diag((hi - lo) / 2)
    hu = jac @ hx @ jac
    assert bool(jnp.all(mask))
    np.testing.assert_allclose(y, s @ hu.T, rtol=2e-4, atol=2e-4 * float(jnp.abs(y).max()))
    # Physical-frame H_x would be wrong here because the box widths differ.
    assert float(jnp.abs(y - s @ hx.T).max()) > 1e-1


def test_diagonal_quadratic_recovers_diagonal_secant_curvature_ratios():
    weights = jnp.array([1.0, 10.0, 100.0])
    lo, hi = -jnp.ones(3), jnp.ones(3)

    def f(x):
        return 0.5 * jnp.sum(weights * x**2)

    xs = jax.random.uniform(jax.random.PRNGKey(2), (5, 3), minval=-1, maxval=1)
    h = _fill(f, xs, lo, hi, 4)
    feats = history_features(h, _current_gu(f, xs[-1], lo, hi))
    ratio = jnp.sinh(4.0 * feats[:, IDX["diag_curvature"]])
    np.testing.assert_allclose(ratio / ratio[0], weights / weights[0], rtol=1e-3)


def test_history_cannot_distinguish_hessians_agreeing_on_observed_span():
    """Rank-deficient history identifies curvature only on span(S)."""
    d = 4
    lo, hi = -jnp.ones(d), jnp.ones(d)
    basis, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(3), (d, d)))
    observed, unobserved = basis[:, :2], basis[:, 2:]
    h1 = _spd(jax.random.PRNGKey(4), d, 10.0)
    # Add curvature only on directions the trajectory never visits, then restore
    # symmetry so that H2 s = H1 s for every s in span(observed).
    p_unobs = unobserved @ unobserved.T
    p_obs = jnp.eye(d) - p_unobs
    h2 = p_obs @ h1 @ p_obs + p_obs @ h1 @ p_unobs + p_unobs @ h1 @ p_obs
    h2 = h2 + p_unobs @ (50.0 * h1) @ p_unobs
    center = jnp.zeros(d)
    coeffs = jnp.array([[0.3, 0.1], [0.1, 0.4], [-0.2, 0.2]])
    xs = center + coeffs @ observed.T

    def make(hm):
        return lambda x: 0.5 * (x - center) @ hm @ (x - center)

    f1, f2 = make(h1), make(h2)
    feats1 = history_features(_fill(f1, xs, lo, hi, 2), _current_gu(f1, xs[-1], lo, hi))
    feats2 = history_features(_fill(f2, xs, lo, hi, 2), _current_gu(f2, xs[-1], lo, hi))
    assert float(jnp.abs(h1 - h2).max()) > 1.0
    np.testing.assert_allclose(feats1, feats2, atol=1e-5)
    s, _, _ = secant_pairs(_fill(f1, xs, lo, hi, 2))
    assert int(np.linalg.matrix_rank(np.asarray(s), tol=1e-6)) == 2 < d


def test_history_features_invariant_to_objective_scale_and_box_rescaling():
    d = 3
    hx = _spd(jax.random.PRNGKey(5), d)
    lo, hi = -2.0 * jnp.ones(d), 2.0 * jnp.ones(d)

    def f(x):
        return 0.5 * x @ hx @ x + 0.1 * jnp.sum(x**3)

    xs = jax.random.uniform(jax.random.PRNGKey(6), (5, d), minval=-2, maxval=2)
    base = history_features(_fill(f, xs, lo, hi, 4), _current_gu(f, xs[-1], lo, hi))

    def scaled(x):
        return 7.5 * f(x)

    obj = history_features(_fill(scaled, xs, lo, hi, 4), _current_gu(scaled, xs[-1], lo, hi))
    c = 3.0

    def boxed(x):
        return f(x / c)

    box = history_features(
        _fill(boxed, c * xs, c * lo, c * hi, 4), _current_gu(boxed, c * xs[-1], c * lo, c * hi)
    )
    np.testing.assert_allclose(obj, base, atol=2e-5)
    np.testing.assert_allclose(box, base, atol=2e-5)


def test_coordinate_permutation_is_equivariant():
    d = 5
    hx = _spd(jax.random.PRNGKey(7), d)
    lo, hi = -jnp.ones(d), jnp.ones(d)

    def f(x):
        return 0.5 * x @ hx @ x

    xs = jax.random.uniform(jax.random.PRNGKey(8), (5, d), minval=-1, maxval=1)
    perm = jnp.array([3, 0, 4, 1, 2])
    hp = hx[perm][:, perm]

    def fp(x):
        return 0.5 * x @ hp @ x

    base = history_features(_fill(f, xs, lo, hi, 4), _current_gu(f, xs[-1], lo, hi))
    permuted = history_features(
        _fill(fp, xs[:, perm], lo, hi, 4), _current_gu(fp, xs[-1, perm], lo, hi)
    )
    np.testing.assert_allclose(permuted, base[perm], atol=1e-5)


def test_negative_curvature_is_reported_not_hidden():
    weights = jnp.array([1.0, -2.0, 3.0])
    lo, hi = -jnp.ones(3), jnp.ones(3)

    def f(x):
        return 0.5 * jnp.sum(weights * x**2)

    xs = jnp.array([[0.1, 0.1, 0.1], [0.1, 0.3, 0.1], [0.3, 0.3, 0.1]])
    h = _fill(f, xs, lo, hi, 2)
    s, y, mask = secant_pairs(h)
    assert bool(jnp.all(mask))
    assert float(s[0] @ y[0]) < 0 < float(s[1] @ y[1])
    feats = history_features(h, _current_gu(f, xs[-1], lo, hi))
    assert float(feats[0, IDX["negative_fraction"]]) == pytest.approx(0.5)
    curvature = feats[:, IDX["diag_curvature"]]
    assert float(curvature[1]) < 0 < float(curvature[0])


def test_repeated_and_roundoff_pairs_are_masked():
    lo, hi = -jnp.ones(2), jnp.ones(2)

    def f(x):
        return jnp.sum(x**2)

    xs = jnp.array([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2 + 1e-9], [0.5, -0.1]])
    _, _, mask = secant_pairs(_fill(f, xs, lo, hi, 3))
    assert mask.tolist() == [False, False, True]


def test_empty_history_masks_features_and_flags():
    h = empty_history(4, 3)
    feats = history_features(h, jnp.array([1.0, -2.0, 0.5]))
    assert feats.shape == (3, HISTORY_FEATURES)
    assert bool(jnp.all(jnp.isfinite(feats)))
    assert bool(jnp.all(feats == 0.0))
    single = push_history(h, jnp.zeros(3), jnp.ones(3), jnp.asarray(1.0))
    feats = history_features(single, jnp.ones(3))
    assert float(feats[0, IDX["has_history"]]) == 0.0
    assert float(feats[0, IDX["valid_fraction"]]) == 0.0


def test_zero_pair_history_is_the_matched_no_history_control():
    h = empty_history(0, 3)
    h = push_history(h, jnp.zeros(3), jnp.ones(3), jnp.asarray(1.0))
    h = push_history(h, jnp.ones(3) * 0.1, jnp.ones(3) * 2, jnp.asarray(0.5))
    s, _, mask = secant_pairs(h)
    assert s.shape == (0, 3) and mask.shape == (0,)
    feats = history_features(h, jnp.ones(3))
    assert feats.shape == (3, HISTORY_FEATURES) and bool(jnp.all(feats == 0.0))


def test_rejection_counts_without_pushing_and_resets_on_acceptance():
    h = push_history(empty_history(2, 2), jnp.zeros(2), jnp.ones(2), jnp.asarray(1.0))
    rejected = record_rejection(record_rejection(h))
    assert int(rejected.rejections) == 2
    np.testing.assert_array_equal(rejected.u, h.u)
    np.testing.assert_array_equal(rejected.valid, h.valid)
    feats = history_features(rejected, jnp.ones(2))
    assert float(feats[0, IDX["rejections"]]) == pytest.approx(np.tanh(0.5))
    accepted = push_history(rejected, jnp.ones(2) * 0.1, jnp.ones(2), jnp.asarray(0.5))
    assert int(accepted.rejections) == 0


def test_nonfinite_records_are_masked_and_features_stay_finite():
    h = empty_history(2, 2)
    h = push_history(h, jnp.zeros(2), jnp.ones(2), jnp.asarray(1.0))
    h = push_history(h, jnp.array([0.1, 0.2]), jnp.array([jnp.nan, 1.0]), jnp.asarray(0.5))
    h = push_history(h, jnp.array([0.3, 0.1]), jnp.array([2.0, 1.0]), jnp.asarray(0.2))
    _, _, mask = secant_pairs(h)
    assert mask.tolist() == [False, False]
    assert bool(jnp.all(jnp.isfinite(history_features(h, jnp.ones(2)))))


def test_quadratic_consistency_is_zero_on_quadratics_and_not_on_cubics():
    lo, hi = -jnp.ones(2), jnp.ones(2)
    xs = jnp.array([[0.1, 0.2], [0.5, -0.3], [-0.4, 0.6]])

    def quad(x):
        return x[0] ** 2 + 3 * x[1] ** 2 + x[0] * x[1]

    def cubic(x):
        return quad(x) + 2 * x[0] ** 3

    fq = history_features(_fill(quad, xs, lo, hi, 2), _current_gu(quad, xs[-1], lo, hi))
    fc = history_features(_fill(cubic, xs, lo, hi, 2), _current_gu(cubic, xs[-1], lo, hi))
    assert abs(float(fq[0, IDX["quadratic_consistency"]])) < 1e-4
    assert abs(float(fc[0, IDX["quadratic_consistency"]])) > 1e-2


def test_global_features_are_broadcast_and_one_dimensional_problems_are_finite():
    def f(x):
        return jnp.sum((x - 0.3) ** 2)

    xs = jnp.array([[0.0], [0.5], [0.1]])
    lo, hi = -jnp.ones(1), jnp.ones(1)
    feats = history_features(_fill(f, xs, lo, hi, 2), _current_gu(f, xs[-1], lo, hi))
    assert feats.shape == (1, HISTORY_FEATURES) and bool(jnp.all(jnp.isfinite(feats)))
    d = 4
    xs4 = jax.random.uniform(jax.random.PRNGKey(9), (4, d), minval=-1, maxval=1)
    feats4 = history_features(
        _fill(f, xs4, -jnp.ones(d), jnp.ones(d), 3),
        _current_gu(f, xs4[-1], -jnp.ones(d), jnp.ones(d)),
    )
    for name in GLOBAL:
        column = feats4[:, IDX[name]]
        np.testing.assert_allclose(column, column[0])


def test_history_ops_jit_vmap_and_differentiate():
    d, pairs = 3, 4
    hx = _spd(jax.random.PRNGKey(10), d)

    def f(x):
        return 0.5 * x @ hx @ x

    lo, hi = -jnp.ones(d), jnp.ones(d)
    batches = [
        _fill(f, jax.random.uniform(jax.random.PRNGKey(20 + i), (5, d), minval=-1, maxval=1),
              lo, hi, pairs)
        for i in range(3)
    ]
    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *batches)
    gu = jnp.ones((3, d))
    batched = jax.vmap(history_features)(stacked, gu)
    assert batched.shape == (3, d, HISTORY_FEATURES)
    np.testing.assert_allclose(
        batched[1], jax.jit(history_features)(batches[1], gu[1]), atol=1e-6
    )
    pushed = jax.jit(push_history)(batches[0], jnp.zeros(d), jnp.ones(d), jnp.asarray(0.0))
    assert pushed.u.shape == (pairs + 1, d)
    grad = jax.grad(lambda g: jnp.sum(history_features(batches[0], g) ** 2))(jnp.ones(d))
    assert bool(jnp.all(jnp.isfinite(grad)))
    grad_hist = jax.grad(lambda u: jnp.sum(history_features(batches[0]._replace(u=u), gu[0])))(
        batches[0].u
    )
    assert bool(jnp.all(jnp.isfinite(grad_hist)))


# --- Task 3: versioned model schema and checkpoints -------------------------

from neuralopt.model import HISTORY_SCHEMA, ModelConfig, init_mlp_model, init_model  # noqa: E402
from neuralopt.optimizer import _validate_policy_params  # noqa: E402
from neuralopt.training import load_checkpoint, save_checkpoint  # noqa: E402

HISTORY_CONFIG = ModelConfig(
    width=16,
    heads=4,
    rank=2,
    layers=1,
    scale_aware=True,
    geometry_bound=10.0,
    lowrank_scale=0.5,
    feature_set="history-v1",
    history_length=4,
)


def test_history_config_sets_input_width_for_both_architectures():
    assert ModelConfig().input_features == 5
    assert HISTORY_CONFIG.input_features == 5 + HISTORY_FEATURES
    params = init_model(jax.random.PRNGKey(0), HISTORY_CONFIG)
    assert params["embed"].shape == (17, 16)
    mlp = init_mlp_model(jax.random.PRNGKey(0), HISTORY_CONFIG)
    assert mlp["hidden"].shape == (19, 16)
    assert init_model(jax.random.PRNGKey(0))["embed"].shape == (5, 32)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_length": 4},
        {"feature_set": "history-v2"},
        {"feature_set": "history-v1", "history_length": -1},
        {"feature_set": "history-v1", "scale_aware": False},
    ],
)
def test_invalid_feature_schema_configurations_raise(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_history_checkpoint_roundtrip_carries_schema(tmp_path):
    params = init_model(jax.random.PRNGKey(1), HISTORY_CONFIG)
    path = tmp_path / "history.npz"
    save_checkpoint(path, params, HISTORY_CONFIG, {"seed": 1})
    with np.load(path) as data:
        assert str(data["schema"][0]) == HISTORY_SCHEMA
    loaded, config = load_checkpoint(path)
    assert config == HISTORY_CONFIG
    np.testing.assert_array_equal(loaded["embed"], params["embed"])


def test_legacy_checkpoint_files_gain_no_schema_key(tmp_path):
    config = ModelConfig(width=8, heads=2, layers=1)
    path = tmp_path / "legacy.npz"
    save_checkpoint(path, init_model(jax.random.PRNGKey(2), config), config)
    with np.load(path) as data:
        assert "schema" not in data.files
        assert data["config"].shape == (7,)


def test_unknown_or_unversioned_extended_schema_fails_closed(tmp_path):
    params = init_model(jax.random.PRNGKey(3), HISTORY_CONFIG)
    path = tmp_path / "history.npz"
    save_checkpoint(path, params, HISTORY_CONFIG)
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    np.savez(tmp_path / "future.npz", **{**arrays, "schema": np.array(["neuralopt-policy/v9"])})
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(tmp_path / "future.npz")
    del arrays["schema"]
    np.savez(tmp_path / "stripped.npz", **arrays)
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(tmp_path / "stripped.npz")


def test_legacy_parameters_are_rejected_for_history_configuration():
    legacy = init_model(jax.random.PRNGKey(4), ModelConfig(width=16, heads=4, layers=1))
    with pytest.raises(ValueError, match="input width"):
        _validate_policy_params(legacy, "transformer", HISTORY_CONFIG)
    _validate_policy_params(init_model(jax.random.PRNGKey(4), HISTORY_CONFIG), "transformer",
                            HISTORY_CONFIG)


# --- Task 4: shared preconditioned evaluation loop ---------------------------

from neuralopt import Problem, minimize  # noqa: E402
from neuralopt.features import denormalize_coordinates  # noqa: E402


def _rotated_problem(d=4, seed=0, budget=60):
    lo = jnp.linspace(-3.0, -1.0, d)
    hi = jnp.linspace(1.0, 6.0, d)
    hx = _spd(jax.random.PRNGKey(seed), d, 50.0)
    center = 0.5 * (lo + hi) + 0.3

    def objective(x):
        return 0.5 * (x - center) @ hx @ (x - center)

    return Problem(objective, (lo, hi), d, max_evaluations=budget)


def _history_params(pairs=8, seed=0):
    config = HISTORY_CONFIG.__class__(**{**HISTORY_CONFIG.__dict__, "history_length": pairs})
    return init_model(jax.random.PRNGKey(seed), config), config


class _Counted:
    def __init__(self, d):
        self.matrix = np.diag(np.geomspace(1.0, 20.0, d))
        self.objective_calls = 0
        self.gradient_calls = 0

    def objective(self, x):
        self.objective_calls += 1
        x = np.asarray(x, dtype=float)
        return float(0.5 * (x - 0.4) @ self.matrix @ (x - 0.4))

    def gradient(self, x):
        self.gradient_calls += 1
        return jnp.asarray(self.matrix @ (np.asarray(x, dtype=float) - 0.4), jnp.float32)


@pytest.mark.parametrize("geometry", ["fixed", "learned"])
def test_preconditioned_loop_counts_every_objective_and_gradient_call(geometry):
    d, budget = 3, 41
    counted = _Counted(d)
    problem = Problem(counted.objective, (-1.0, 1.0), d, gradient=counted.gradient)
    params, config = _history_params(4)
    result = minimize(
        problem,
        max_evaluations=budget,
        params=params,
        config=config,
        candidate_method="preconditioned",
        hamiltonian_metric=geometry,
        fixed_metric_diag=np.ones(d),
        preconditioned_step=0.2,
    )
    diag = result.diagnostics
    assert diag["objective_evaluations"] == counted.objective_calls
    assert diag["gradient_evaluations"] == counted.gradient_calls
    assert diag["hvp_evaluations"] == 0
    total = counted.objective_calls + counted.gradient_calls
    assert result.evaluations == total and budget - 1 <= total <= budget
    assert counted.gradient_calls == 1 + diag["accepted_steps"]


def test_history_records_are_coherent_evaluated_states():
    problem = _rotated_problem(4, budget=30)
    params, config = _history_params(16)
    result = minimize(
        problem,
        params=params,
        config=config,
        candidate_method="preconditioned",
        record_diagnostics=True,
    )
    hist = result.diagnostics["history"]
    lo, hi = problem.bounds
    valid = np.asarray(hist["valid"])
    assert valid.sum() == 1 + result.diagnostics["accepted_steps"] >= 2
    for u, g, f in zip(
        np.asarray(hist["u"])[valid], np.asarray(hist["g"])[valid], np.asarray(hist["f"])[valid],
        strict=True,
    ):
        x = denormalize_coordinates(jnp.asarray(u), lo, hi)
        assert bool(jnp.all((x >= lo) & (x <= hi)))
        np.testing.assert_allclose(f, problem.objective(x), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            g, gradient_to_normalized(jax.grad(problem.objective)(x), lo, hi), rtol=1e-4,
            atol=1e-4,
        )
    assert result.fun <= float(problem.objective(problem.x0))


def test_all_rejected_candidates_terminate_without_pushing_history():
    d, budget = 2, 20

    def objective(x):
        return 1.0

    def gradient(x):
        return jnp.ones(d)

    problem = Problem(objective, (-1.0, 1.0), d, gradient=gradient)
    params, config = _history_params(4)
    result = minimize(
        problem,
        max_evaluations=budget,
        params=params,
        config=config,
        candidate_method="preconditioned",
        record_diagnostics=True,
    )
    diag = result.diagnostics
    assert diag["accepted_steps"] == 0
    # A candidate is only proposed while an accepted gradient could still be paid
    # for (used + 2 <= budget), so one unit stays unspent when every step fails.
    assert diag["rejected_steps"] == budget - 3 == diag["objective_evaluations"] - 1
    assert result.evaluations == budget - 1
    assert int(np.sum(diag["history"]["valid"])) == 1
    assert diag["history"]["rejections"] == budget - 3


def test_clipped_candidates_are_the_evaluated_and_stored_points():
    d = 3
    lo, hi = -np.ones(d), np.ones(d)

    def objective(x):
        return jnp.sum((x - 4.0) ** 2)  # optimum outside the box

    problem = Problem(objective, (lo, hi), d)
    result = minimize(
        problem,
        max_evaluations=20,
        candidate_method="preconditioned",
        hamiltonian_metric="fixed",
        fixed_metric_diag=np.ones(d),
        preconditioned_step=10.0,
        record_diagnostics=True,
    )
    for record in result.diagnostics["records"]:
        c = np.asarray(record["candidate"])
        assert np.all(c >= lo) and np.all(c <= hi)
        np.testing.assert_allclose(record["candidate_objective"], objective(jnp.asarray(c)),
                                   rtol=1e-6)
    np.testing.assert_allclose(result.x, hi, atol=1e-6)


def test_nonfinite_candidates_are_rejected_and_never_stored():
    d = 2

    def objective(x):
        return jnp.where(x[0] > 0.3, jnp.nan, jnp.sum((x - 0.9) ** 2))

    problem = Problem(objective, (-1.0, 1.0), d)
    params, config = _history_params(8)
    result = minimize(
        problem,
        max_evaluations=30,
        params=params,
        config=config,
        candidate_method="preconditioned",
        record_diagnostics=True,
    )
    hist = result.diagnostics["history"]
    valid = np.asarray(hist["valid"])
    assert np.all(np.isfinite(np.asarray(hist["f"])[valid]))
    assert np.isfinite(result.fun)
    assert any(not np.isfinite(r["candidate_objective"]) for r in result.diagnostics["records"])


def test_preconditioned_loop_validation_errors():
    problem = _rotated_problem(2)
    with pytest.raises(ValueError, match="preconditioned_step"):
        minimize(problem, candidate_method="preconditioned", hamiltonian_metric="fixed",
                 fixed_metric_diag=np.ones(2))
    params, config = _history_params(4)
    for method in ("direct", "langevin", "underdamped", "hamiltonian"):
        with pytest.raises(ValueError, match="preconditioned"):
            minimize(problem, params=params, config=config, candidate_method=method)


@pytest.mark.parametrize("d", [2, 4, 8, 10])
def test_learned_history_policy_runs_across_dimensions(d):
    problem = _rotated_problem(d, seed=d, budget=40)
    params, config = _history_params(4, seed=d)
    result = minimize(problem, params=params, config=config, candidate_method="preconditioned")
    assert np.isfinite(result.fun) and result.fun <= float(problem.objective(problem.x0))
    assert result.diagnostics["candidate_method"] == "preconditioned"


# --- Task 5: history meta-training -------------------------------------------

from neuralopt.model import init_mlp_model as _init_mlp  # noqa: E402
from neuralopt.training import (  # noqa: E402
    HistoryUnrollConfig,
    history_regret_loss,
    history_unroll,
    train,
)


def _known(problem):
    return Problem(
        problem.objective, problem.bounds, problem.dimension, metadata={"known_optimum": 0.0}
    )


LEGACY_SMALL = ModelConfig(width=16, heads=4, rank=2, layers=1, scale_aware=True,
                           geometry_bound=10.0, lowrank_scale=0.5)


@pytest.mark.parametrize(
    ("model_kind", "config"),
    [("transformer", HISTORY_CONFIG), ("mlp", HISTORY_CONFIG), ("transformer", LEGACY_SMALL)],
)
def test_history_regret_loss_has_finite_nonzero_meta_gradient(model_kind, config):
    problem = _known(_rotated_problem(3))
    init = init_model if model_kind == "transformer" else _init_mlp
    params = init(jax.random.PRNGKey(0), config)
    unroll = HistoryUnrollConfig(steps=6, horizons=(2, 6))
    loss, grads = jax.jit(
        jax.value_and_grad(
            lambda p, k: history_regret_loss(p, problem, k, config, unroll, model_kind)
        )
    )(params, jax.random.PRNGKey(1))
    leaves = jax.tree_util.tree_leaves(grads)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves) > 0


def test_history_unroll_config_validation():
    with pytest.raises(ValueError):
        HistoryUnrollConfig(steps=4, horizons=(8,))
    with pytest.raises(ValueError):
        HistoryUnrollConfig(candidate_weight=-1.0)


def test_training_unroll_matches_inference_loop_rule():
    problem = _known(_rotated_problem(3, seed=2))
    params, config = _history_params(4, seed=3)
    steps = 8
    unroll = HistoryUnrollConfig(steps=steps, horizons=(steps,))
    rolled = history_unroll(
        params, problem, jax.random.PRNGKey(0), config, unroll, x0=problem.x0
    )
    result = minimize(
        problem,
        max_evaluations=2 + 2 * steps,
        params=params,
        config=config,
        candidate_method="preconditioned",
    )
    inference = np.asarray(result.trajectory[1 : steps + 1])
    np.testing.assert_allclose(np.asarray(rolled["values"])[: inference.size], inference,
                               rtol=1e-4, atol=1e-6)
    assert int(np.sum(rolled["accepted"][: inference.size])) == sum(
        a < b for a, b in zip(inference, [result.trajectory[0], *inference[:-1]], strict=True)
    )


def test_train_supports_history_objective():
    problems = [_known(_rotated_problem(2, seed=s)) for s in range(2)]
    params, losses = train(
        problems,
        steps=3,
        config=HISTORY_CONFIG,
        objective="history",
        history_unroll=HistoryUnrollConfig(steps=4, horizons=(2, 4)),
    )
    assert len(losses) == 3 and all(np.isfinite(losses))
    assert params["embed"].shape == (17, 16)


# --- Review fixes ------------------------------------------------------------


def _history_arrays(tmp_path):
    params = init_model(jax.random.PRNGKey(5), HISTORY_CONFIG)
    path = tmp_path / "history.npz"
    save_checkpoint(path, params, HISTORY_CONFIG)
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize("mutation", ["feature_code", "fractional_length", "stripped_truncated"])
def test_malformed_history_checkpoints_fail_closed(tmp_path, mutation):
    arrays = _history_arrays(tmp_path)
    config = arrays["config"].copy()
    if mutation == "feature_code":
        config[7] = 2.0
    elif mutation == "fractional_length":
        config[8] = 4.5
    else:
        del arrays["schema"]
        config = config[:7]
    arrays["config"] = config
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "bad.npz")


def test_recorded_step_length_is_the_applied_length():
    d = 2
    problem = Problem(lambda x: 1.0, (-1.0, 1.0), d, gradient=lambda x: jnp.ones(d))
    result = minimize(
        problem,
        max_evaluations=8,
        candidate_method="preconditioned",
        hamiltonian_metric="fixed",
        fixed_metric_diag=np.ones(d),
        preconditioned_step=0.2,
        record_diagnostics=True,
    )
    lengths = [r["step_length"] for r in result.diagnostics["records"]]
    np.testing.assert_allclose(lengths, [0.2, 0.1, 0.05, 0.025, 0.0125][: len(lengths)])


def test_nonfinite_learned_geometry_fails_closed():
    params, config = _history_params(4)
    params = {**params, "diag": params["diag"] * jnp.nan}
    with pytest.raises(FloatingPointError, match="geometry"):
        minimize(_rotated_problem(3), params=params, config=config,
                 candidate_method="preconditioned")


def test_training_scan_history_equals_inference_history():
    problem = _known(_rotated_problem(3, seed=2))
    params, config = _history_params(4, seed=3)
    steps = 8
    rolled = history_unroll(params, problem, jax.random.PRNGKey(0), config,
                            HistoryUnrollConfig(steps=steps, horizons=(steps,)), x0=problem.x0)
    accepted = int(np.sum(rolled["accepted"]))
    # This budget admits exactly `steps` candidates when both loops agree.
    result = minimize(problem, max_evaluations=2 + steps + accepted + 1, params=params,
                      config=config, candidate_method="preconditioned", record_diagnostics=True)
    assert len(result.trajectory) == steps + 1
    final = rolled["final_history"]
    inference = result.diagnostics["history"]
    np.testing.assert_array_equal(np.asarray(final.valid), np.asarray(inference["valid"]))
    for field in ("u", "g", "f"):
        np.testing.assert_allclose(np.asarray(getattr(final, field)),
                                   np.asarray(inference[field]), rtol=1e-4, atol=1e-6)
    assert int(final.rejections) == inference["rejections"]
