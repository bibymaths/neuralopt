# Outcome-driven training

`unrolled_regret_loss` uses `jax.lax.scan` to differentiate projected learned
updates. At configured horizons it minimizes robust normalized best-so-far
regret, `log1p(max(f_best-f*, 0) / max(|f_initial-f*|, 1))`. It also includes
small step, ascent-direction, and log-diagonal regularizers. There is no claimed
gradient through hard inference acceptance or certified branching.
Regret training requires an explicit `problem.metadata["known_optimum"]`; it
never assumes that an untranslated objective has optimum zero. All differentiable
normalizations use square-root smoothing so their derivatives remain finite at
stationary points.
Directional regularization transforms physical gradients into normalized box
coordinates before comparing them with the learned direction, so anisotropic
parameter widths cannot reverse the ascent/descent label.
The imitation target uses the same normalized-coordinate gradient; no training
objective now compares policy directions with gradients from a different
coordinate system.

Training uses explicit PRNG keys, Optax global-norm clipping, nonfinite loss
detection, and portable checkpoints. The test suite verifies finite, nonzero
meta-gradients.

The first CPU smoke run used 12 meta-updates, 10 unrolled steps, horizons
2/5/10, and dimensions 2 and 4. It took 39.69 seconds. Loss rose from 0.2901 to
0.7010, and the checkpoint worsened an unseen 8D rotated quadratic from 89.41
to 164.71. The artifact is retained as `neuralopt-v0.2.0-smoke.npz` but is not
promoted or bundled. This rejects the hypothesis that this tiny training budget
is sufficient; validation-based early stopping and a substantially larger,
family-separated curriculum remain required.

Per-task loss-and-gradient functions are now JIT compiled. Repeating the same
12-update, eight-task smoke configuration reduced training time from 39.69 to
21.10 seconds; a 20-update single-task run took 2.39 seconds. The remaining cost
is dominated by first-use compilation for every distinct objective closure.
Once compiled, a 10-step meta-gradient evaluation drops from about 1.69 seconds
in eager mode to roughly 1.3 milliseconds on the same task. Grouping compatible
tasks into batched, parameterized objective families is the principal training
throughput opportunity.

## Stochastic geometry training

`stochastic_geometry_loss` differentiates through explicit-noise overdamped
Euler or fixed-mass BAOAB trajectories. At every episode the transformer sees
the current point, gradient, incumbent, and remaining budget, predicts a
strictly positive diagonal plus low-rank factor, and freezes it for the episode.
The next episode may predict different geometry, so the complete training
policy is adaptive and has no fixed Gibbs invariant-measure guarantee.
BAOAB training initializes momentum from `N(0, T M)` using the first predicted
mass, matching inference and retaining the pathwise mass/noise gradient.

The primary loss is weighted multi-horizon `log1p` best-feasible regret. A
normalized displacement penalty discourages unstable trajectories. Integration
length is static, so the explicit objective/gradient compute term is constant
and deliberately has zero parameter gradient; there is no differentiable early
stopping surrogate. Best-so-far `minimum` and incumbent selection are
nondifferentiable at ties and use JAX subgradients. Hard inference validity
checks, certification branches, and benchmark timing do not lie on the training
gradient path.

For overdamped training, a quadratic test function estimates the conditional
weak generator identity for each episode-frozen mobility. Its Gaussian noise
contribution is integrated analytically; sampling its square would add variance
that spuriously rewards collapsing the mobility. The squared residual is passed
through `log1p` before weighting because raw residual scales differ by orders of
magnitude across task families. This checks the implemented Euler generator; it
neither removes finite-step bias nor makes the history-adaptive policy
stationary. The optional entropy term fits a Gaussian to trajectory states and
is explicitly a free-energy surrogate, not exact differential entropy or
thermodynamic free energy. It was disabled in the v0.4 reported experiment, and
its dense covariance/log-determinant path is skipped completely when disabled.

The reproducible configuration and checkpoints are described in
`distribution-aware-training.md`.
