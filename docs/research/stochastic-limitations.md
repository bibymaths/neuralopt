# Stochastic research limitations and gate record

## Gate status

- **A — theory/dependencies:** pass for native equations and dependency audit.
  Diffrax is optional; Exponax is deferred for lack of a justified PDE use.
- **B — overdamped Langevin:** pass for fixed/episode-frozen geometry and
  numerical tests; optimization evidence is negative overall.
- **C — learned geometry:** partial. Existing learned geometry executes, but no
  Langevin-specific checkpoint exists and identity is competitive or better.
- **D — underdamped dynamics:** pass for fixed-mass BAOAB numerics and smoke
  comparison; learned-mass superiority is unsupported.
- **E — distribution-aware training:** partial. Exact divergence, weak generator,
  and Gaussian free energy exist; transformer training ablations are unimplemented.
- **F — function space:** partial. Periodic reference geometry refines correctly;
  learned cross-resolution transfer is untested.
- **G — certification:** blocked. Stochastic incumbent injection is not integrated;
  certification and replay remain unchanged and passing.
- **H — repository validation:** pending the final v0.3 validation pass.

## Mathematical limitations

Euler–Maruyama and BAOAB are biased at finite step size. The bounded logistic
map excludes exact boundary states from stochastic dynamics, although the
incumbent archive preserves boundary optima. Exact mobility divergence scales
poorly with dimension. Weak residuals use Monte Carlo estimates and finite-time
differences. History-adaptive transformer geometry has no fixed invariant target.

The function-space module validates a discretized reference problem, not an
infinite-dimensional learned optimizer. No mesh-independent transformer claim,
global convergence claim, or stochastic certification claim is supported.

The smoke study is too small for publication-level confidence intervals and its
hyperparameters are not tuned on a separate validation distribution. All new
stochastic methods remain opt-in.
