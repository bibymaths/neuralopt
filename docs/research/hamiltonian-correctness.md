# Hamiltonian correctness and evidence

## Implemented guarantees and restrictions

- The fixed metric is positive diagonal plus low rank and is constant during a
  transition.
- Momentum covariance and kinetic energy use the same mass convention.
- Leapfrog uses `lax.scan`, explicit keys, momentum reversal, and log-space
  Metropolis correction.
- Box constraints use a differentiable interior transform with its Jacobian.
- The unconstrained chain coordinate is retained exactly between transitions;
  transform clipping is initialization-only.
- Nonfinite energy and excessive energy error mark a divergent proposal.
- General nonlinear constraints and host-only objectives fail closed.
- Adaptive learned metrics have no stationary-HMC claim.

Tests compare Woodbury solves and kinetic energy against dense references,
estimate momentum covariance from 20,000 draws, reverse leapfrog trajectories,
check reduced drift at smaller step size, test interior feasibility, invalid
metrics, and preservation of an exact boundary incumbent. These numerical tests
support but do not replace the standard leapfrog reversibility and
volume-preservation argument.

## Acceptance diagnosis and smoke experiment

`scripts/run_hamiltonian_smoke.py` evaluates 15 matched instances with a nominal
100-objective-evaluation budget. Direct NeuralOpt proposes 99 candidates; each
Hamiltonian variant performs 16 transitions, 32 leapfrog steps, 97 accounted
objective evaluations, and 64 gradient evaluations. Results are in
`results/hamiltonian-smoke-v0.2.0.json`.

| Variant | Transition acceptance | Objective improvement | Incumbent update | Warm runtime |
|---|---:|---:|---:|---:|
| Direct NeuralOpt | 1.01% | 1.01% | 1.01% | 0.245 s |
| Identity HMC | 93.75% | 50.0% | 31.25% | 0.092 s |
| Learned-metric HMC | 93.75% | 56.25% | 18.75% | 0.083 s |

Identity HMC beats direct final objective on 8/15 paired cases; learned metric
beats direct on 9/15 and identity on only 7/15. Median objective ratios versus
direct are 0.771 and 0.745, respectively, but objectives from different families
must not be pooled as an absolute accuracy statistic. HMC helps rotated
quadratics, Rastrigin, and Ackley in this smoke sample, but is substantially
worse on isotropic and ill-conditioned quadratics.

Median absolute energy error is 0.114 for identity and 0.162 for learned metric.
The first identity-HMC call is about 1.06 s because it compiles the transition;
warm timings reuse compilation. High HMC acceptance therefore does not establish
better optimization or end-to-end cold-start efficiency.

## Hypothesis status

- **H1:** provisionally supported only on selected families; rejected as a
  universal statement.
- **H2:** not supported; learned geometry does not consistently beat identity.
- **H3:** provisionally supported by distinct exploration improvements and a
  monotone incumbent, but needs larger multimodal studies.
- **H4:** unresolved. Warm runtime is favorable under this implementation, but
  gradient information and cold compilation differ materially.
- **H5:** untested. Certification code is unchanged and all replay tests pass,
  but Hamiltonian incumbents are not yet injected into certified search.

The evidence is a smoke experiment with three seeds, not a powered benchmark.
Hamiltonian proposals remain opt-in and must not replace the direct default.
