"""Synthetic two-parameter kinetic inverse problem using differentiable RK4."""

import jax
import jax.numpy as jnp

from neuralopt import Problem, minimize


def simulate(theta, times, y0=1.0):
    k1, k2 = theta

    def step(y, dt):
        def f(z):
            return -k1 * z / (k2 + z)

        a = f(y)
        b = f(y + dt * a / 2)
        c = f(y + dt * b / 2)
        d = f(y + dt * c)
        yn = y + dt * (a + 2 * b + 2 * c + d) / 6
        return yn, yn

    _, ys = jax.lax.scan(step, jnp.asarray(y0), jnp.diff(times))
    return jnp.concatenate((jnp.array([y0]), ys))


def main():
    times = jnp.linspace(0, 4, 25)
    truth = jnp.array([0.8, 0.35])
    observed = simulate(truth, times) + 0.005 * jax.random.normal(
        jax.random.PRNGKey(7), times.shape
    )

    def residual(p):
        return simulate(p, times) - observed

    def objective(p):
        return jnp.mean(residual(p) ** 2)

    p = Problem(
        objective,
        (jnp.array([0.05, 0.05]), jnp.array([2.0, 2.0])),
        2,
        x0=jnp.array([1.5, 1.5]),
        residual=residual,
        max_evaluations=250,
        identifier="saturating-kinetics",
    )
    result = minimize(p, seed=1)
    print("truth", truth, "estimate", result.x, "MSE", result.fun)
    print("A low residual does not imply unique identification; k1 and k2 are correlated.")


if __name__ == "__main__":
    main()
