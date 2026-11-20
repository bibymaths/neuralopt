# v0.5 transferable learned geometry

## Result

The prespecified hypothesis was not supported. Under matched objective budgets,
the corrected transformer did not outperform a validation-tuned fixed diagonal
on the 30-instance held-out set. This negative result is retained because it
identifies a representation limitation that lower training loss obscures.

## Reconciliation of v0.4

Re-evaluating the original checkpoints, seeds, and configurations reproduced
every deterministic reported metric exactly; timings were intentionally excluded
from this equality check. The machine-readable comparison is
`results/geometry-diagnostics-v0.5.0.json`.

The two reported overdamped FP-minus-regret values came from different checkpoint
generations, not different evaluation populations. Commit `c9ee283` reported
`-0.089 [-0.181, -0.008]` on ten pairs. After review fixes removed sampled
quadratic-noise variance from the FP loss and matched BAOAB initial momentum to
inference, commit `0ecc22f` retrained the checkpoints and reported
`-0.048 [-0.105, -0.006]` on the same ten pairs. The corrected value supersedes
the pre-review result.

## Diagnosis and minimal corrections

The v0.4 feature map normalized the gradient direction and discarded its norm:
scaling a gradient by 1000 changed the predicted diagonal by only `1.2e-7`.
Consequently the model could not transfer across objective scales. v0.5 reuses
the same input width but replaces the constant feature, when enabled, with a
bounded log gradient-scale feature. The corresponding checkpoint changes its
diagonal by `0.131` in the same diagnostic.

The positive-definite geometry remains

\[A_\theta=\operatorname{diag}(d_\theta)+U_\theta U_\theta^\top.\]

For new checkpoints, `d` is bounded symmetrically in log space and `U` is
dimension/rank normalized and bounded with `tanh`. This controls condition and
low-rank energy without changing legacy checkpoint behavior. Median learned
condition was `1.23` and the largest per-run median was `3.09`. Diagonal-only,
low-rank-only, and full heads are explicit ablations. Temperature and friction
are sampled from their inference ranges, and multiple explicit random keys can
be averaged for pathwise stochastic meta-gradients.

Four-sample gradients did not reduce observed gradient-norm variability in this
small diagnostic (coefficient of variation `2.39`, versus `1.78` with one
sample). This is evidence against assuming that simple key averaging solves the
variance problem; it also mixes task heterogeneity and is not an estimator of
within-task variance alone. The fixed-compute loss remains constant and has no
gradient. Incumbent `min` decisions are piecewise differentiable; gradients flow
through the selected branch but not through the discrete selection boundary.

The remaining dominant limitation is information, not positive definiteness.
At a step, the transformer sees position, one gradient, bounds, incumbent, and
scale. A single gradient cannot identify rotated curvature. Episode updates add
history dependence but do not expose curvature history as a sufficient statistic.
The model therefore improves easy isotropic cases while regressing on the
ill-conditioned and rotated families. No architectural expansion was made
because the held-out result did not justify it.

## Protocol

- Unchanged small transformer: width 16, four heads, rank two, one layer.
- Ten training and ten validation tasks; 30 final tasks with disjoint seeds and
  entirely unseen dimensions 7, 8, and 10.
- Isotropic, condition-1000, rotated condition-100, Rastrigin, and Ackley
  families; dimensions 2--10; objective scales 0.2--5 and coordinate scales
  0.5--2.
- Training ranges: temperature 0.25--1, friction 0.5--2; horizons 2/4/8;
  144 updates. A 36-update full checkpoint tests training budget.
- Fixed scalar selected only on validation from 0.25, 0.5, 1, 2, and 4.
  Selected values were 0.25 for overdamped and 2 for BAOAB.
- Final budget 100 with paired seeds and 10,000 paired bootstrap resamples.

Reproduce training and evaluation with
`uv run python scripts/run_transfer_geometry_v0_5.py`; pass
`--reuse-checkpoints` to re-evaluate saved checkpoints. Reproduce the diagnosis
with `uv run python scripts/run_geometry_diagnostics_v0_5.py`.

## Held-out comparisons

Differences below are paired mean `log1p(regret)` relative to tuned fixed
geometry; negative is better.

| Dynamics and geometry | Mean | Difference (95% CI) | Wins/30 |
|---|---:|---:|---:|
| Overdamped fixed | 3.207 | 0 | -- |
| Overdamped learned diagonal | 3.282 | +0.075 [-0.064, 0.246] | 13 |
| Overdamped learned full | 3.295 | +0.088 [0.021, 0.164] | 10 |
| Overdamped learned FP | 3.286 | +0.079 [0.009, 0.178] | 10 |
| Overdamped learned low rank | 3.522 | +0.315 [0.065, 0.609] | 7 |
| BAOAB fixed | 2.869 | 0 | -- |
| BAOAB learned diagonal | 2.922 | +0.053 [-0.247, 0.353] | 14 |
| BAOAB learned full | 2.927 | +0.058 [-0.204, 0.322] | 13 |

The full overdamped model at 36 updates was already worse than fixed
(`+0.103 [-0.058, 0.329]`); 144 updates narrowed uncertainty but remained worse.
This rejects insufficient training budget as the sole explanation. FP
regularization did not establish an improvement over fixed or regret-only
training.

Family results explain the aggregate. Overdamped full geometry improved the
isotropic mean (0.686 versus 0.745), but regressed on ill-conditioned quadratics
(7.031 versus 6.835) and rotated quadratics (4.052 versus 3.954). Low-rank-only
was especially weak on those families. BAOAB learned geometry helped isotropic
tasks but lost on ill-conditioned, rotated, and Rastrigin tasks. Thus no favorable
aggregate is used to conceal family regressions.

Frozen-checkpoint transfer at temperatures 0.125 and 1.5 and budgets 50 and 150
also found no credible improvement. Overdamped point estimates were worse in all
12 learned comparisons; at temperature 1.5 and budget 150 all three learned
intervals excluded zero on the harmful side. BAOAB intervals crossed zero in all
eight comparisons. These settings were not used to retune the fixed baseline.

All methods used the same objective budget and recorded objective/gradient
evaluations, stochastic integration steps, success, time-to-target, incumbent
and candidate improvements, displacement, condition, finite rate, first-call
time, and warm time in `results/transfer-geometry-v0.5.0.json`. Training time and
checkpoint size are recorded there as well. Timings are single-host descriptive
measurements because compilation caches make process-order comparisons imperfect.
Existing direct and identity-HMC results remain available in
`results/hamiltonian-smoke-v0.2.0.json`; they are contextual baselines rather
than part of this geometry ablation because they use a different transition and
historical task population. They were not pooled with the matched v0.5 pairs.

## Mathematical scope and limitations

Overdamped noise uses covariance `2 T h A`; BAOAB uses the exact OU covariance
`T(1-exp(-2 gamma h)) M`. Tests cover positive definiteness, low-rank algebra,
fluctuation--dissipation scaling, stochastic reproducibility, pathwise gradient
flow, weak generator identities, and Gaussian equilibrium. The FP term is
skipped entirely when disabled, avoiding overflow in an inactive diagnostic.

The learned policy is episode-frozen only within each rollout segment and is
history-dependent between segments. It is therefore an adaptive optimization
policy, not fixed Gibbs-invariant dynamics. Euler and BAOAB also have finite-step
bias; no exact finite-step stationarity is claimed. Certification and independent
certificate replay are unchanged, and neural or stochastic candidates never
replace a rigorous certificate.

Only one meta-training seed and 30 final instances were affordable in this
study. Per-family samples are six, so family means are diagnostic rather than
high-powered. Objective closures are compiled independently, and transformer
overhead remains material. A future test of learned curvature should first add a
compact, explicitly justified secant/history statistic and preregister a larger
multi-seed evaluation; it should not enlarge the transformer by default.
