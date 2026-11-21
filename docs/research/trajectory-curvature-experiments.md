# Prespecified trajectory-curvature experiment protocol (draft — approve before runs)

## Primary contrasts
1. v0.5 single observation vs. no-attention history summary.
2. No-attention history summary vs. matched-parameter tiny trajectory transformer.
3. Learned trajectory geometry vs. validation-tuned fixed-diagonal geometry (primary scientific comparator).
4. Learned trajectory geometry vs. L-BFGS/self-scaled Broyden numerical references where appropriate (benchmarks only).

## Dataset split
- Use distinct train, validation and **new** final test seeds and task families, recording generator versions and hashes. Do not recycle v0.5's final test results as an optimization target.
- Train across 2/4/8 dimensions (provisional) and test at 7/10/16 or similarly predeclared unseen dimensions; add rotated condition numbers 10, 100, 1000, with parameter and objective scale perturbations.
- Families: isotropic SPD, diagonal anisotropy, rotated SPD, indefinite smooth quadratics with bounded domains, Rosenbrock, Rastrigin/Ackley, noisy gradients, optional inexpensive inverse problem.
- For exact curvature identification tasks, evaluate known Hessian on sampled span, report relative secant residual and rank of observed S, not whole-matrix error beyond observed subspace.

## Models and controls
- Frozen v0.5 transformer; validation-tuned fixed diagonal; history statistics without attention; matched MLP; history attention K=4,8; random-history/masked-history; optional JAX HVP probes; historical BFGS/SR1 reference (outside inference).
- Match training compute, model size where meaningful, history initialization, objective/gradient/HVP counts and candidate budgets.
- Separate direct optimizer and overdamped/BAOAB evaluations; don't pool incompatible reference populations.

## Primary metrics
- Paired mean `log1p(regret)` minus fixed-diagonal, 95% paired bootstrap CI (negative better), wins per pair, per-family and outlier analysis.
- Time-to-target and anytime regret, nonfinite/failure rate, gradient-direction alignment, incumbent updates, candidate acceptance, preconditioner conditioning, secant residual/rank and history diversity.
- Total obj/grad/HVP/integration operations, latency of history encoding, first JIT compilation, warm runtime, peak memory and meta-training time/checkpoint size.

## Required statistics
- At least 3 independent meta-training seeds (budget permitting), paired independent task instances; report effect sizes and uncertainty with hierarchical or seed-aware resampling rather than inflating N by repeated trajectories of the same checkpoint.
- Predeclare primary endpoint, success threshold and stop criterion. Controlled ablation before architectural expansion; reports must show failures and uncertainty.

## Gates and falsification
- **Information gate:** a synthetic rotated SPD problem with sufficient independent secants must exhibit correct `y=Hs` and informative subspace curvature; reject features failing this.
- **Compatibility gate:** old checkpoints produce unchanged outputs in legacy mode; fail if not.
- **Model-value gate:** history model must improve regret on fresh held-out rotated/ill-conditioned tasks vs no-history; otherwise stop and revise features.
- **Attention gate:** attention must improve over no-attention history at matched cost; otherwise do not keep it.
- **Scientific gate:** learned model must beat **tuned fixed diagonal** in at least one preregistered domain with credible cost-adjusted advantage and no concealed major reliability regression. If not, retain fixed geometry as default.
- **Certification gate:** native certification/replay unchanged and tests pass.

## Artifact layout
Version new outputs under `results/trajectory-curvature-v0.6.0*.json` and `artifacts/trajectory-v0.6.0-*.npz`; scripts must reproduce figures from raw results and never rewrite v0.5 evidence. Training checkpoints must embed schema, task splits and RNG provenance.
