# v0.4 distribution-aware geometry experiment

## Equations and gradient path

The transformer predicts `A = diag(d) + U U^T`, where `d` is produced by a
softplus plus `1e-4`. Overdamped training uses

\[z_{n+1}=z_n-hA\nabla U(z_n)+\sqrt{2Th}\,(D^{1/2}\xi+U\eta),\]

with the bounded transformed potential `U(z) = f(x(z)) - T log|dx/dz|`.
BAOAB uses the same factor as a mass matrix, its Woodbury inverse in the drift,
and exact Ornstein--Uhlenbeck momentum covariance
`T(1-exp(-2 gamma h)) M`. Explicit PRNG keys make trajectories reproducible and
pathwise differentiation carries gradients through both drift geometry and
noise scale.

The weak Fokker--Planck term compares the antithetic one-step change of
`phi(z)=mean(z^2)` with

\[L\phi=b^T\nabla\phi+\tfrac12\operatorname{tr}(a\nabla^2\phi),
\qquad a=2TA.\]

Mobility is frozen only within an episode. Updating it from trajectory history
makes the optimizer nonstationary. Euler and BAOAB retain finite-step bias; the
experiment makes no exact Gibbs or finite-step stationarity claim.

## Protocol

- Model: width 16, four heads, rank two, one transformer layer.
- Training: 36 Adam updates at `3e-4`, global gradient clipping, seed 17.
- Trajectory: six episodes, two integration steps per episode, horizons 2/4/6,
  step size 0.008, temperature 0.5.
- Training set: shifted, rotated, condition-10 to condition-1000 quadratics,
  Rastrigin, Ackley, and Rosenbrock in dimensions two through four.
- Held out: ten paired instances spanning rotated and ill-conditioned
  quadratics, Rosenbrock, Ackley, and Rastrigin in unseen dimensions three to
  six and seeds 40/41.
- Primary evaluation: objective budget 80 and temperature 0.5. Paired
  percentile bootstrap intervals use 10,000 resamples.
- Transfer evaluation: temperatures 0.25/1.0 and budgets 40/120 on four held-out
  instances. These small four-instance intervals are descriptive.

The machine-readable record is
`results/geometry-training-v0.4.0.json`. The three portable checkpoints are in
`artifacts/neuralopt-v0.4.0-*.npz`. Reproduce with
`uv run python scripts/run_geometry_training_v0_4.py`; use
`--reuse-checkpoints` to repeat evaluation without retraining.

## Primary held-out results

Differences are paired mean `log1p(regret)` relative to identity; negative is
better. All variants used the same objective budgets. Overdamped runs used 78
gradient evaluations and 52 integration steps; BAOAB used 75 and 30.

| Integrator and geometry | Mean log regret | Difference vs identity (95% CI) | Wins/10 |
|---|---:|---:|---:|
| Overdamped identity | 2.854 | 0.000 | -- |
| Overdamped fixed diagonal | 2.804 | -0.050 [-0.282, 0.143] | 3 |
| Overdamped untrained transformer | 2.661 | -0.193 [-0.553, 0.075] | 5 |
| Overdamped regret trained | 2.723 | -0.132 [-0.427, 0.112] | 5 |
| Overdamped regret + FP | 2.634 | -0.220 [-0.574, 0.025] | 5 |
| BAOAB identity | 1.712 | 0.000 | -- |
| BAOAB fixed diagonal | 1.272 | -0.441 [-0.878, -0.078] | 6 |
| BAOAB untrained transformer | 1.460 | -0.253 [-0.790, 0.091] | 4 |
| BAOAB regret trained | 1.532 | -0.181 [-0.716, 0.197] | 4 |
| BAOAB transferred regret + FP | 1.648 | -0.064 [-0.312, 0.149] | 3 |

The direct paired FP-minus-regret comparison is `-0.089` with 95% interval
`[-0.181, -0.008]` for overdamped dynamics (6/10 wins). This is evidence on the
small prespecified held-out set, not a universal advantage. For BAOAB transfer
the difference is `+0.116 [-0.323, 0.771]`, so no improvement is established.

At unseen temperatures and budgets the overdamped FP-minus-regret point estimate
remained negative in all four settings, but every four-instance interval touched
or crossed zero. At budget 40, transferred FP geometry was worse for BAOAB at
both temperatures (`+0.885` at 0.25 and `+0.871` at 1.0). The simple fixed
diagonal remains the strongest supported BAOAB result.

Training took 27.9 seconds for overdamped regret, 26.3 seconds for BAOAB regret,
and 24.2 seconds for regret plus FP on the recorded CPU run. Median warm
inference was about 0.08--0.12 seconds. First-call timings in the shared process
are cache- and order-sensitive and are retained in the artifact rather than
interpreted as isolated compilation measurements.

## Correctness and remaining gaps

Tests verify positive diagonals, diagonal-plus-low-rank covariance, exact OU
fluctuation--dissipation scaling, reproducibility, Gaussian moments, finite
nonzero gradients into both geometry heads, weak generator identities, and
nonfinite-coordinate rejection. Certification code and independent replay are
unchanged.

The experiment is still small: six training tasks, ten primary held-out pairs,
one training seed, and no BBOB or scientific inverse problem. FP weighting was
not selected on a separate validation sweep, free-energy regularization remains
disabled, and objective closures compile independently. Stronger claims require
multiple training seeds, larger held-out suites, family-level uncertainty, and
batched parameterized task generators.
