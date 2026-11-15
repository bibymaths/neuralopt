# Hamiltonian-informed optimization

## Mathematical facts

For fixed inverse temperature `beta` and mass matrix `M`, NeuralOpt defines the
artificial density `pi(x) proportional to exp(-beta f(x))` and Hamiltonian

\[H(x,p)=\beta f(x)+\tfrac12p^T M^{-1}p,\qquad p\sim N(0,M).\]

The implementation uses velocity `M^-1 p`, not `M p`. Its learned convention is
explicit: the transformer output `D + U U^T` is the **mass matrix**. Momentum is
sampled as `sqrt(D) z + U eta`, and Woodbury algebra applies the inverse without
forming a dense matrix.

Leapfrog alternates a half momentum update, full position update, and half
momentum update. For a fixed separable Hamiltonian it is reversible (with final
momentum reversal), symplectic, and volume preserving. The Metropolis decision
is evaluated in log space: `log(u) < min(0, H_initial - H_proposed)`. These are
the standard constructions described by the Oxford and Washington lecture
notes, Hoffman and Gelman, Betancourt, and the Stan reference manual.

Box constraints use `x = lo + (hi-lo) sigmoid(z)`. The unconstrained potential
is

\[U(z)=\beta f(x(z))-\log|dx/dz|.\]

The Jacobian term is required for the transformed target. Exact boundary points
are not represented by finite `z`; the optimization incumbent is therefore a
separate feasible archive that retains and can return boundary points.

## Optimization versus sampling

Metropolis acceptance means that a phase-space proposal is valid for the fixed
target under the documented assumptions. It does **not** mean the objective
improved. An accepted transition may move uphill, while a rejected proposal may
still contain a better feasible point worth archiving for optimization. NeuralOpt
records transition acceptance, improvement over the exploration state, and
best-incumbent updates separately.

The learned metric is fixed during one transition but may change between
transitions with optimizer history. This is an adaptive optimization heuristic;
no stationary-distribution or detailed-balance claim is made for that adaptive
chain. The fixed identity-metric kernel is the reference sampling transition.

## Executable example

```python
import jax
import jax.numpy as jnp
from neuralopt import Problem, init_model, minimize
from neuralopt.hamiltonian import HamiltonianConfig

problem = Problem(lambda x: jnp.sum((x - 0.3) ** 2), (-2.0, 2.0), 4)
result = minimize(
    problem,
    params=init_model(jax.random.PRNGKey(0)),
    candidate_method="hamiltonian",
    hamiltonian_metric="identity",
    hamiltonian_config=HamiltonianConfig(step_size=0.03, leapfrog_steps=2),
    max_evaluations=100,
    record_diagnostics=True,
)
```

## Primary references

- Deligiannidis, *Advanced Simulation Methods*, Lecture 7, Oxford.
- Chen, *Hamiltonian Monte Carlo*, STAT 535 Lecture 9, Washington.
- Stan Reference Manual, “MCMC Sampling”.
- Betancourt (2017), “A Conceptual Introduction to Hamiltonian Monte Carlo,”
  arXiv:1701.02434.
- Hoffman and Gelman (2014), “The No-U-Turn Sampler,” *JMLR* 15:1593–1623,
  arXiv:1111.4246.
