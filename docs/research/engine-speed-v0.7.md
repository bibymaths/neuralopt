# v0.7 Track A: compiled inference and training (speed only)

Track A changes **how** the v0.6 `preconditioned` rule runs, not what it computes. All
new paths are opt-in.

- `minimize(..., preconditioned_engine="compiled")` and
  `neuralopt.compiled.preconditioned_solve(_batch)` give the same results as the Python
  loop: identical to 1e-10 in float64, and statistically equivalent in float32.
- The default stays `"python"`, so the frozen v0.6 results and the sha256 of their script
  remain reproducible bit for bit.

## A1. Where the time went (profile)

One warm tf-h8 solve: frozen v0.6 checkpoint, rotated κ=100, d=16, budget 100, float32.

| Component | ms per solve |
|---|---:|
| `push_history` (38 eager calls, 1.9 ms each) | 71 |
| objective + gradient, including **2 recompilations per call** (`minimize` rebuilds `jit(grad(f))` every time) | 64 |
| eager coordinate transforms (normalize, candidate, gradient→u, denormalize) | 59 |
| jitted features + model (120 dispatch-bound calls, ~140 µs each) | 17 |
| remaining Python, eager ops and host syncs | 43 |
| **total (instrumented; 241 ms uninstrumented)** | 256 |
| estimated compiled compute, vmap-amortized | **0.8 (0.3%)** |

Host dispatch and per-call recompilation account for more than 99% of the wall time.

## A2–A4. Compiled engine

- **Loop.** A single `jit` runs a `lax.while_loop`. The gradient is evaluated under
  `lax.cond` only when a candidate improves, so budget accounting is unchanged. Errors
  come back as status codes and are raised after the loop. Diagnostics go into
  preallocated length-`budget` arrays.
- **Batching.** `preconditioned_solve_batch` vmaps over tasks `objective(x, theta)` of
  one family and dimension. Under `vmap`, `lax.cond` evaluates both branches, but
  gradients are still charged only on acceptance; this is tested.
- **Parity.** Float64 parity is tested on rotated and Rastrigin problems across five
  arms: history transformer, history MLP, legacy transformer, fixed and identity.
  Trajectory, stored history, records and counts match to 1e-10.
- **float32.** On 12 rotated tasks with the frozen tf-h8 checkpoint:
  - accepted-step counts were identical on all 12;
  - final log10 regret differed by a mean of +0.0000 decades (median |D| 3.4e-5,
    maximum 0.001);
  - trajectories differ only at float32 rounding level.
- **Recompilation.** Repeated solves with new weights of the same shapes do not
  recompile.

## A5. Training

- **One compilation per family and dimension.** `neuralopt.task_families` reproduces the
  v0.6 generator exactly (tested to 1e-10), with parametric objectives `f(x, theta)`.
- **One compiled training step.** `neuralopt.fast_training.train_batched` compiles the
  loss, the meta-gradient and the Optax update together, vmapped over a batch of tasks.
- **NaN guard.** A step with a nonfinite loss or meta-gradient leaves parameters and
  optimizer state unchanged. It is counted in `TrainingReport.skipped_steps`, never
  dropped silently. This is tested with `f(x) = Σ√(1+x)`, which is singular at the
  lower box edge.

## A6. Benchmark (`results/engine-benchmark-v0.7.0.json`)

Conditions:
- Single process per arm, float64, CPU.
- 32 rotated κ=100 quadratics, d=16, with interior optima.
- Budget 100 objective+gradient units.
- **Cold** is the first solve in a fresh process, including compilation.
- **Warm** is the median single solve on new tasks of the same family.
- **Batched** is tasks per second over the 32 tasks.

| Arm | Cold s | Warm ms | Batched tasks/s | Median log10 regret | Obj / grad per solve |
|---|---:|---:|---:|---:|---:|
| tf-h8, Python loop | 2.13 | 297.8 | 4.1 (sequential) | −2.72 | 62.8 / 36.4 |
| tf-h8, compiled | 0.83 | 4.08 | 453 (vmap) | −2.72 | 62.8 / 36.4 |
| fixed step, compiled | 0.56 | 1.04 | 4,698 (vmap) | −2.88 | 54.6 / 44.7 |
| SciPy L-BFGS-B (jitted value_and_grad) | 0.09 | 9.84 | 102 (sequential) | −11.19 | 50 / 50 |
| Optimistix BFGS | 0.46 | 0.17 | 24,636 (vmap) | −1.21 | 50 / 50 |
| Optimistix LBFGS | 0.71 | 0.22 | 12,031 (vmap) | −9.25 | 50 / 50 |

Training on the 21 v0.6 training tasks (float64, history-v1 K=8, 24-step unroll):

| Trainer | Compilations | Compile s | Task-gradients/s after compile |
|---|---:|---:|---:|
| v0.6 `train()` (one closure per task) | 21 | 36.5 | 13.1 |
| `train_batched`, batch 1 | 6 | ≈16 (first pass) | 122 |
| `train_batched`, batch 8 | 6 | ≈8.5 (first pass) | 350 |

How to read the tables:
- **Speed.** The compiled engine is about 73× faster per warm solve than the Python
  loop, and about 2.4× faster than SciPy L-BFGS-B. Batched, it solves about 110× more
  tasks per second than the Python loop. It remains about 20× slower than Optimistix
  per solve.
- **Quality is unchanged.** Track A does not change what is computed: learned
  geometry's regret is still far behind L-BFGS.
- **Optimistix BFGS** used defaults with no tuning. It charges one value+gradient per
  step, line-search trials included. It reached only −1.21 here and is reported as
  measured.
- **Optimistix has no box bounds.** Its solvers are unbounded; with interior optima the
  box is inactive at the solution, but iterates may leave it. Optimistix is a benchmark
  dependency only (`neuralopt[benchmark]`) and never runs inside NeuralOpt inference.

Reproduce: `uv run python scripts/benchmark_engines_v0_7.py`.
