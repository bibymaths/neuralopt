# Compiled engine

Since v0.7.0, `minimize(..., candidate_method="preconditioned")` runs on the compiled
engine by default (`preconditioned_engine="compiled"`): the whole loop is one
`lax.while_loop` inside one `jit`. It computes the same rule as the Python loop
(float64 parity to 1e-10) and is much faster; it does not change what the optimizer
achieves.

```python
result = minimize(problem, max_evaluations=100, candidate_method="preconditioned")
```

## Requirements

- The objective must be JAX-traceable. A problem with a host/NumPy `gradient=` raises a
  `ValueError`; pass `preconditioned_engine="python"` for it.
- Reuse the same objective function object to reuse the compilation.

## Batched solves

`neuralopt.compiled.preconditioned_solve_batch` `vmap`s the loop over tasks
`objective(x, theta_i)` of one family and dimension; `neuralopt.task_families` provides
parametric families. Under `vmap`, both branches of the acceptance `lax.cond` are
computed, but gradients are charged only on acceptance.

## Speed (v0.7 benchmark)

One warm learned (tf-h8) solve took 296.1 ms on the Python loop and 4.00 ms compiled
(about 74×). See the [engine-speed report](../research/engine-speed-v0.7.md) for the
equal-work benchmark against SciPy and Optimistix, cold and batched timings, and the
caveats.
