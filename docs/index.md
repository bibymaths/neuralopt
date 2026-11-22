<p align="center">
  <img src="assets/images/logo.png" alt="NeuralOpt" width="420">
</p>

# NeuralOpt

**A JAX research package for learned optimization, with replayable interval
certification.**

!!! warning "Status: alpha research package"
    NeuralOpt is for research on learned optimizers. In our own pre-registered and
    exploratory studies (v0.4–v0.7), the learned geometries did **not** beat
    validation-tuned fixed or Barzilai–Borwein baselines, and classical L-BFGS
    outperforms the learned optimizer on the reported benchmarks. See
    [When to use NeuralOpt](when-to-use.md) and the [release notes](release-notes/index.md).

NeuralOpt has two independent modes:

- **Learned mode.** A compact transformer reads normalized position, gradient, bounds
  and budget features (and, opt-in, a masked history of evaluated states) and proposes
  the next step. No conventional optimizer runs during inference, and every objective
  and gradient evaluation is counted against one budget.
- **Certified mode.** Interval branch-and-bound with outward rounding proves global
  bounds for a restricted expression language. It uses no learned component, and its
  certificates can be replayed by an independent verifier.

## Where to start

- [Installation](installation.md) and [Quickstart](quickstart.md)
- [When to use NeuralOpt](when-to-use.md) (and when to use SciPy or Optimistix)
- User guide: [learned mode](guide/learned-mode.md),
  [certified mode](guide/certified-mode.md), [compiled engine](guide/compiled-engine.md)
- [Research reports](research/gram-space-v0.7.md), [release notes](release-notes/index.md),
  [changelog](changelog.md) and [limitations](limitations.md)
- [Citing NeuralOpt](citing.md)
