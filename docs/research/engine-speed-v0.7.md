# v0.7 Track A: compiled inference and training (speed only)

Track A changes **how** the v0.6 `preconditioned` rule runs, not what it computes. All
new paths are opt-in.

- `minimize(..., preconditioned_engine="compiled")` and
  `neuralopt.compiled.preconditioned_solve(_batch)` give the same results as the Python
  loop: identical to 1e-10 in float64, and statistically equivalent in float32.
- **Since the v0.7.0 release the default is `"compiled"`** (see "Default engine" below).
  The v0.6 reproduction script pins `"python"`, so the frozen v0.6 results remain
  reproducible bit for bit.

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

## A6. Benchmark: equal work (`results/engine-benchmark-v0.7.0.json`)

### Conditions
- Single process per arm, float64, CPU.
- 32 rotated κ=100 quadratics, d=16, with interior optima.
- Every arm gets **100 units** (objective plus gradient calls).
- An iteration is one candidate for the NeuralOpt loop: 1 unit, plus 1 if the candidate
  improved and its gradient was taken. For L-BFGS-B and Optimistix it is one
  objective+gradient call: 2 units.
- **Cold** is the first solve in a fresh process, including compilation.
- **Warm** is the median single solve on new tasks of the same family.
- **Batched** is tasks per second over all 32 tasks.

### How each arm's budget is enforced
- **NeuralOpt loop:** it proposes a candidate only while `used + 2 ≤ 100`.
- **SciPy L-BFGS-B:** the objective function raises after 50 calls. Line-search calls
  count, because every SciPy call passes through it.
- **Optimistix:** `max_steps = 50`. Each step makes exactly one `fn` evaluation with its
  linearization (2 units), whether the line search accepts or rejects the trial point.
  - A call counter confirmed calls = steps for both BFGS and LBFGS.
  - Optimistix's own convergence test can stop it early; it never did here.
  - Reporting its final value requires one extra `fn(sol.value)`. That is measurement,
    not part of the solve.

### Results

| Arm | Units used | Iterations | ms / iteration | Warm ms | Cold s | Batched tasks/s | Median log10 regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| tf-h8, Python loop | 99.2 | 61.8 | 4.873 | 296.1 | 2.10 | 4.2 (sequential) | −2.72 |
| tf-h8, compiled | 99.2 | 61.8 | 0.066 | 4.00 | 0.84 | 467 (vmap) | −2.72 |
| fixed step, compiled | 99.3 | 53.6 | 0.021 | 1.08 | 0.57 | 4,385 (vmap) | −2.88 |
| SciPy L-BFGS-B | 100 | 50 | 0.195 | 9.77 | 0.08 | 101 (sequential) | −11.19 |
| Optimistix BFGS | 100 | 50 | 0.003 | 0.14 | 0.45 | 25,405 (vmap) | −1.21 |
| Optimistix LBFGS | 100 | 50 | 0.004 | 0.22 | 0.72 | 11,997 (vmap) | −9.25 |

### Time to target

The estimated time to target is the median number of iterations to reach the target
multiplied by the warm time per iteration.

| Arm | Reached 1e-3 | Units | Iterations | Est. ms | Reached 1e-6 | Units | Iterations | Est. ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tf-h8, Python loop | 10/32 | 83 | 51 | 248.5 | 0/32 | — | — | not reached |
| tf-h8, compiled | 10/32 | 83 | 51 | 3.35 | 0/32 | — | — | not reached |
| fixed step, compiled | 11/32 | 87 | 43 | 0.88 | 0/32 | — | — | not reached |
| SciPy L-BFGS-B | 32/32 | 26 | 13 | 2.54 | 32/32 | 56 | 28 | 5.47 |
| Optimistix BFGS | 0/32 | — | — | not reached | 0/32 | — | — | not reached |
| Optimistix LBFGS | 32/32 | 39 | 19.5 | 0.085 | 32/32 | 70 | 35 | 0.153 |

### Why Optimistix BFGS reaches only −1.21

It is not a tolerance problem.
- With `rtol = atol = 1e-14` (the A6 run) and again with `1e-8`, every one of the 32
  tasks used all 50 steps and ended with result code 7, "maximum number of steps
  reached". The median regret was −1.21 both times.
- On task 0 the iterate moved on only 8 of the 50 steps. The other 42 were backtracking
  line-search trials, each charged one evaluation.
- Its default initial inverse Hessian is the identity, which overshoots badly on these
  κ=100 problems. LBFGS scales its initial Hessian from the secant pair and reaches
  −9.25.
- Neither solver was tuned beyond these two tolerance settings.

### Where the compiled tf-h8 time goes (warm solve ≈ 4.2 ms, 61 iterations)

| Component | Time |
|---|---:|
| fixed per-call overhead (budget-2 solve: initial f, gradient, history, dispatch, host transfer) | 1.07 ms |
| per iteration | ≈ 46–51 µs |
| · model forward, features included (measured inside one compiled loop) | 27.9 µs |
| · objective + gradient (gradient on 58% of iterations) | 0.18 µs |
| · remaining loop machinery (history push, records, larger loop state) | ≈ 18–23 µs |
| **model floor** (forward × iterations) | **1.70 ms per solve** |

For comparison, the fixed-step compiled loop costs 0.76 µs per iteration plus about
1.0 ms of per-call overhead. The learned arm is therefore limited by the model forward
pass and the history-related loop state, not by objective work.

### Training (21 v0.6 training tasks; float64; history-v1 K=8; 24-step unroll)

| Trainer | Compilations | Compile s | Task-gradients/s after compile |
|---|---:|---:|---:|
| v0.6 `train()` (one closure per task) | 21 | ≈36 | ≈13 |
| `train_batched`, batch 1 | 6 | ≈16 (first pass) | ≈120 |
| `train_batched`, batch 8 | 6 | ≈8.5 (first pass) | ≈350 |

### How to read these numbers

- **Speed.** The compiled engine makes the learned policy about 74× faster per solve
  than the Python loop. Per solve it is about 2.4× faster than SciPy L-BFGS-B, and still
  about 18× slower than Optimistix LBFGS.
- **Quality is unchanged.** Track A does not change what is computed: under equal work,
  learned geometry reaches 1e-3 on 10 of 32 tasks and 1e-6 on none, while L-BFGS-B and
  Optimistix LBFGS reach 1e-6 on all 32.
- **Optimistix is a benchmark dependency only** (`neuralopt[benchmark]`) and never runs
  inside NeuralOpt inference. Its solvers have no box bounds; with interior optima the
  box is inactive at the solution, but iterates may leave it.

## Default engine

- While v0.7 was in development, `preconditioned_engine` defaulted to `"python"`.
- **At the v0.7.0 release the default flipped to `"compiled"`**, as planned here.
  Objectives with a host/NumPy `gradient=` are not traceable; with the default engine
  they now raise a `ValueError` naming `preconditioned_engine="python"` (fail closed,
  not a silent fallback).
- In the same change, `scripts/run_trajectory_curvature_v0_6.py` passes
  `preconditioned_engine="python"` explicitly at both preconditioned `minimize` calls,
  and a test checks the pin. The frozen v0.6 results are unchanged.
- The script's sha256 therefore changed. It is recorded here rather than by rewriting
  the frozen results file:
  - recorded in `results/trajectory-curvature-v0.6.0.json`:
    `35c1e401e6bc8bccfaecbb9168708335e9a7d4fbd6498e35d635ed54da5e50d1`;
  - from v0.7.0: `f80e8acc6bda802d65a00ea73e80e179ffcf7f8d0ddd9e23492220581b27e9f3`.

Reproduce: `uv run python scripts/benchmark_engines_v0_7.py` (add `--profile` or
`--training` for those sections alone).
