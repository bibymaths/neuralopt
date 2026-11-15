# Gibbs measures and optimization

At temperature `T`, the artificial finite-dimensional Gibbs density is
`pi_T(x) proportional to exp(-f(x)/T)`. It is a distribution over feasible
states, not an optimizer certificate. Lower temperature concentrates mass near
low objective values under suitable assumptions but does not guarantee that a
finite trajectory finds a global minimizer.

The logistic bounded transform includes its determinant so fixed-geometry
continuous-time dynamics target the mapped bounded Gibbs measure. Euler,
BAOAB, and adaptive learned geometry introduce distinct qualifications:

- Euler–Maruyama and BAOAB have finite-step bias.
- Episode-frozen learned geometry is fixed only during one episode.
- Updating geometry from history makes the overall process nonstationary.
- Changing temperature creates a sequence of targets, not one invariant Gibbs
  distribution.

Optimization therefore reports the best feasible incumbent separately from the
stochastic state. Distributional accuracy, candidate improvement, and incumbent
updates are independent measurements.
