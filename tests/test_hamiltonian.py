import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralopt import Problem, init_model, minimize
from neuralopt.hamiltonian import (
    HamiltonianConfig,
    from_unconstrained,
    hamiltonian_proposal,
    kinetic_energy,
    leapfrog,
    mass_solve,
    sample_momentum,
    to_unconstrained,
    validate_metric,
)


def test_mass_algebra_and_momentum_covariance():
    diag = jnp.array([0.7, 1.3])
    lowrank = jnp.array([[0.4], [-0.2]])
    dense = np.diag(np.asarray(diag)) + np.asarray(lowrank) @ np.asarray(lowrank).T
    vector = jnp.array([0.3, -1.1])
    np.testing.assert_allclose(
        mass_solve(vector, diag, lowrank), np.linalg.solve(dense, vector), rtol=2e-6
    )
    np.testing.assert_allclose(
        kinetic_energy(vector, diag, lowrank),
        0.5 * np.asarray(vector) @ np.linalg.solve(dense, vector),
    )
    keys = jax.random.split(jax.random.PRNGKey(4), 20_000)
    samples = jax.vmap(lambda key: sample_momentum(key, diag, lowrank))(keys)
    np.testing.assert_allclose(np.cov(np.asarray(samples), rowvar=False), dense, atol=0.03)


def test_leapfrog_reversibility_and_energy_convergence():
    def potential(q):
        return 0.5 * q @ q

    diag = jnp.array([1.0, 2.0])
    lowrank = jnp.zeros((2, 0))
    q0, p0 = jnp.array([0.3, -0.7]), jnp.array([0.2, 1.1])
    q1, p1 = leapfrog(q0, p0, potential, diag, lowrank, 0.02, 20)
    recovered_q, recovered_p = leapfrog(q1, -p1, potential, diag, lowrank, 0.02, 20)
    np.testing.assert_allclose(recovered_q, q0, atol=2e-6)
    np.testing.assert_allclose(recovered_p, -p0, atol=2e-6)

    def error(step_size):
        q, p = leapfrog(q0, p0, potential, diag, lowrank, step_size, 10)
        initial = potential(q0) + kinetic_energy(p0, diag, lowrank)
        final = potential(q) + kinetic_energy(p, diag, lowrank)
        return abs(float(final - initial))

    assert error(0.025) < error(0.1)

    def phase_map(phase):
        q, p = leapfrog(phase[:2], phase[2:], potential, diag, lowrank, 0.03, 3)
        return jnp.concatenate((q, p))

    jacobian = jax.jacfwd(phase_map)(jnp.concatenate((q0, p0)))
    np.testing.assert_allclose(np.linalg.det(np.asarray(jacobian)), 1.0, atol=2e-5)


def test_bounded_transform_and_hmc_proposal():
    lo, hi = jnp.array([-2.0, 1.0]), jnp.array([3.0, 5.0])
    x = jnp.array([-1.5, 4.5])
    np.testing.assert_allclose(from_unconstrained(to_unconstrained(x, lo, hi), lo, hi), x)

    def objective(point):
        return jnp.sum((point - jnp.array([0.2, 2.0])) ** 2)

    proposal = hamiltonian_proposal(
        jax.random.PRNGKey(2),
        x,
        objective,
        lo,
        hi,
        jnp.ones(2),
        jnp.zeros((2, 0)),
        HamiltonianConfig(step_size=0.02, leapfrog_steps=4),
    )
    assert np.all(np.asarray(proposal.candidate) > np.asarray(lo))
    assert np.all(np.asarray(proposal.candidate) < np.asarray(hi))
    assert np.isfinite(proposal.energy_error)
    assert proposal.objective_evaluations == 8
    assert proposal.gradient_evaluations == 5


def test_invalid_hamiltonian_configuration_and_metric_fail_closed():
    with pytest.raises(ValueError, match="positive"):
        HamiltonianConfig(step_size=0)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_metric(jnp.array([1.0, 0.0]), jnp.zeros((2, 0)))


def test_hamiltonian_optimizer_separates_chain_and_boundary_incumbent():
    lo = jnp.array([-1.0, -2.0])
    problem = Problem(
        lambda x: jnp.sum((x - lo) ** 2),
        (lo, jnp.array([2.0, 3.0])),
        2,
        x0=lo,
        max_evaluations=50,
    )
    result = minimize(
        problem,
        params=init_model(jax.random.PRNGKey(0)),
        candidate_method="hamiltonian",
        hamiltonian_config=HamiltonianConfig(step_size=0.01, leapfrog_steps=4),
        record_diagnostics=True,
    )
    np.testing.assert_array_equal(result.x, lo)
    assert result.fun == 0.0
    assert np.all(np.diff(result.trajectory) <= 0)
    assert result.diagnostics["hmc_transitions"] == 6
    assert result.diagnostics["objective_evaluations"] == 49
    assert result.diagnostics["gradient_evaluations"] == 36
    assert result.diagnostics["incumbent_update_rate"] == 0.0
    assert result.diagnostics["objective_improvement_rate"] >= 0.0
    assert result.diagnostics["hmc_transition_acceptance"] is not None


def test_fixed_hmc_recovers_gaussian_moments():
    def objective(x):
        return 0.5 * jnp.sum(x * x)

    lo, hi = -8.0 * jnp.ones(2), 8.0 * jnp.ones(2)
    state = jnp.zeros(2)
    key = jax.random.PRNGKey(9)
    samples = []
    accepted = 0
    config = HamiltonianConfig(step_size=0.12, leapfrog_steps=5)
    for iteration in range(1_200):
        key, proposal_key = jax.random.split(key)
        proposal = hamiltonian_proposal(
            proposal_key,
            state,
            objective,
            lo,
            hi,
            jnp.ones(2),
            jnp.zeros((2, 0)),
            config,
        )
        state = proposal.state
        accepted += int(proposal.accepted)
        if iteration >= 200:
            samples.append(np.asarray(state))
    draws = np.asarray(samples)
    np.testing.assert_allclose(draws.mean(axis=0), np.zeros(2), atol=0.15)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), np.eye(2), atol=0.2)
    assert accepted / 1_200 > 0.8
