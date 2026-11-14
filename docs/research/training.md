# Outcome-driven training

`unrolled_regret_loss` uses `jax.lax.scan` to differentiate projected learned
updates. At configured horizons it minimizes robust normalized best-so-far
regret, `log1p(max(f_best-f*, 0) / max(|f_initial-f*|, 1))`. It also includes
small step, ascent-direction, and log-diagonal regularizers. There is no claimed
gradient through hard inference acceptance or certified branching.

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
