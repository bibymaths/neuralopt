# Benchmark protocol

Machine-readable rows must separate objective calls, gradient calls, derivative
products, compilation, warm execution, loading, and total time. Compilation is
not silently amortized. Comparisons are paired by instance and seed and report
medians, intervals, success, and anytime regret.
Objective wrappers enforce the requested budget even when a third-party solver
would finish an over-budget line search. Regret is `null`, rather than zero, when
a task has no known optimum.
All baselines return the tracked best finite in-budget observation. Nonfinite
samples cannot replace a valid incumbent; a run with no finite observation fails
explicitly instead of emitting a misleading NaN result.

The current v0.2 artifact is only a smoke experiment. A fixed normalized-gradient
control beats v0.1.0 on every recorded family; for example, its final values are
0.0286 versus 0.4685 (isotropic), 24.61 versus 293.0 (ill-conditioned), and
5.918 versus 71.46 (rotated). These are single instances and do not support a
superiority claim. SciPy, Adam, Optimistix, differential evolution, quasi-random
search, trained MLP, and BBOB performance profiles remain an uncompleted M4
milestone.

## Development performance profile

Single-run CPU timings from the development environment (November 2026) are
diagnostic, not publication benchmarks. Compiling the policy, objective, and
gradient once per run reduced warm learned-inference cost as follows:

| Dimension | Before | Current | Evaluations |
|---:|---:|---:|---:|
| 4 | 7.80 ms/eval | 1.18 ms/eval | 100 |
| 32 | 8.32 ms/eval | 1.70 ms/eval | 100 |

Cold 100-evaluation runs remain about 1.1--1.2 seconds because JAX compilation
dominates. Warm execution still synchronizes once per candidate and uses a
Python optimization loop; batching or a compiled loop is the next runtime
bottleneck. Independently, the measured 0--4.1% v0.1 acceptance rate remains a
larger algorithmic efficiency problem than kernel dispatch.
