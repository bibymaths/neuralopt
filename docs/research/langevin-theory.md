# Langevin theory and implementation

For fixed positive-definite mobility `A`, overdamped Langevin dynamics are

\[dX_t=-A\nabla f(X_t)dt+\sqrt{2TA}\,dW_t.\]

Under regularity and integrability assumptions, the continuous-time invariant
density is proportional to `exp(-f/T)`. NeuralOpt uses Euler–Maruyama with
episode-frozen identity or transformer geometry. The finite-step chain is biased
and is not described as an exact Gibbs sampler. Noise is sampled with covariance
`A = D + U U^T`; tests verify covariance, reproducibility, temperature scaling,
and Gaussian moments.

For position-dependent mobility, the Itô drift compatible with the Gibbs density
contains `T div A`. `mobility_divergence` computes this exactly with `jacfwd` for
small-dimensional experiments. It materializes a rank-three Jacobian and is not
enabled in the production optimizer. History-adaptive geometry does not inherit
a fixed invariant-measure guarantee.

Underdamped dynamics use

\[dX=M^{-1}Pdt,\qquad dP=-\nabla fdt-\gamma Pdt+\sqrt{2\gamma TM}\,dW.\]

The native BAOAB splitting applies half force, half position, exact
Ornstein–Uhlenbeck momentum, half position, and half force updates. Its OU step
uses `exp(-gamma h)` and covariance `T(1-exp(-2 gamma h))M`, satisfying the
fluctuation–dissipation relation. BAOAB also has finite-step invariant-measure
bias.

Box constraints use the same logistic coordinate transform and Jacobian as HMC.
The unconstrained state is threaded between episodes; exact boundary optima are
retained by the separate feasible incumbent.

Primary references include Roberts and Tweedie (1996), Girolami and Calderhead
(2011), Welling and Teh (2011), and the geometric Langevin literature summarized
in `hamiltonian-related-work.md`.

Diffrax 0.7.2 is an optional `sde` extra and supplies an independent identity-
mobility Euler reference. It is not imported by the base package. Exponax was
audited at version 0.2.1 but not added: the present weak-form and periodic-field
experiments do not require a density PDE solver, so adding it would not yet have
a demonstrated numerical purpose.
