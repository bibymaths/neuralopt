# Fokker–Planck foundations

For Itô drift `b` and covariance `a = sigma sigma^T`, the generator is

\[\mathcal L\varphi=b^T\nabla\varphi+
\tfrac12\operatorname{tr}(a\nabla^2\varphi).\]

The forward Kolmogorov equation is the adjoint evolution of this generator. For
fixed mobility Langevin dynamics it becomes

\[\partial_t\rho=\nabla\cdot[A(\rho\nabla f+T\nabla\rho)].\]

NeuralOpt implements the small-dimensional generator and Monte Carlo weak-form
residual

\[(E[\varphi(X_{n+1})]-E[\varphi(X_n)])/h-E[\mathcal L\varphi(X_n)].\]

Tests validate the Ornstein–Uhlenbeck identity with antithetic noise. This is a
diagnostic primitive, not yet a transformer training result. A low residual can
coexist with poor optimization regret and is not itself evidence for the central
learned-optimizer hypothesis.

The free energy `E[f] + T int rho log rho` is implemented exactly only for a
Gaussian distribution and quadratic objective. The test confirms its minimum at
the analytic Gibbs covariance. No arbitrary particle entropy estimator is
labelled exact free energy.

Jordan, Kinderlehrer, and Otto (1998) interpret Fokker–Planck evolution as a
Wasserstein gradient flow through a minimizing-movement scheme. That
continuous-time/variational result is not transferred to finite-step Euler or
history-adaptive transformer coefficients.
