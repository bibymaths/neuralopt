# Stochastic learned-geometry research status

The research gap under investigation is narrower than “neural stochastic
optimization.” Learned optimizers already predict updates and preconditioners;
Riemannian Langevin/HMC and L2HMC already adapt sampling geometry. NeuralOpt asks
whether a variable-dimensional transformer can transfer episode-frozen geometry
across optimization task families while distributional structure regularizes
candidate generation.

No novelty claim is established by the current implementation. The existing
v0.1 transformer was not trained for Langevin mobility, and smoke results show
that its geometry often underperforms identity. Exact divergence and weak-form
utilities create a testable path to Fokker–Planck-informed training, but no such
training ablation has yet demonstrated improved regret.

## v0.3 smoke evidence

The 15-instance, three-seed artifact is
`results/stochastic-smoke-v0.3.0.json`. All variants use a nominal objective-call
budget of 100 and report gradient/integration counts separately.

| Variant | Median objective | Improvement rate | Incumbent rate | Warm time |
|---|---:|---:|---:|---:|
| Direct | 2.912 | 1.01% | 1.01% | 0.235 s |
| Identity HMC | 2.449 | 50.0% | 31.25% | 0.103 s |
| Learned HMC | 2.302 | 56.25% | 18.75% | 0.090 s |
| Identity overdamped | 7.986 | 51.52% | 3.03% | 0.135 s |
| Learned overdamped | 17.449 | 54.55% | 3.03% | 0.139 s |
| Identity BAOAB | 1.970 | 52.63% | 15.79% | 0.119 s |
| Learned BAOAB | 1.937 | 52.63% | 21.05% | 0.109 s |

Overdamped identity and learned mobility beat direct on only 5/15 and 4/15
paired cases. Identity and learned BAOAB each beat direct on 11/15. Learned
geometry has not established a consistent advantage over identity. Pooled
objectives across different families are descriptive only, not a valid absolute
accuracy statistic.
