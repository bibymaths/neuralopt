# Learned mode

`minimize(problem, mode="learned", ...)` (the default mode) runs a learned policy on a
box-bounded problem. Learned mode requires a JAX-differentiable objective or an explicit
`gradient=`.

## Problems

```python
from neuralopt import Problem

problem = Problem(
    objective,            # f(x) -> scalar
    (lower, upper),       # scalars or arrays, finite, lower < upper
    dimension,
    x0=None,              # default: the box centre
    gradient=None,        # optional host gradient (Python engine only, see below)
    max_evaluations=1000,
    dtype=jnp.float32,    # use jnp.float64 (with x64 enabled) for research runs
)
```

## Candidate methods

| `candidate_method` | What proposes the next point |
|---|---|
| `"direct"` (default) | the transformer emits the step directly (v0.1 policy) |
| `"preconditioned"` | step `u' = clip(u − σ L P ĝ, −1, 1)` with a learned, fixed or identity metric; one objective call per candidate, a gradient only on acceptance, σ halves after a rejection (v0.6) |
| `"hamiltonian"` | Hamiltonian proposals with identity, fixed or learned mass (v0.2) |
| `"langevin"`, `"underdamped"` | overdamped Euler–Maruyama or BAOAB trajectories (v0.3–v0.4) |

For `"preconditioned"`, `hamiltonian_metric` selects `"learned"`, `"fixed"` (with
`fixed_metric_diag` and `preconditioned_step`) or `"identity"` geometry.

## Checkpoints

- `params=None` loads the bundled `demo-v0.1.0` smoke checkpoint (legacy 5-feature
  model).
- History models (`ModelConfig(feature_set="history-v1", history_length=K)`) need
  `candidate_method="preconditioned"` and carry a versioned schema; unknown or
  inconsistent schemas fail closed. v0.1–v0.5 checkpoints load identically.
- `model_kind="mlp"` and `model_kind="gram"` need caller-provided parameters; the Gram
  model (v0.7) runs on the compiled engine only.

## Accounting

With `record_diagnostics=True`, `result.diagnostics` reports objective, gradient and HVP
evaluations, accepted and rejected steps, and per-step records. `result.evaluations` is
the total charged to the budget.

See the [research reports](../research/trajectory-curvature-v0.6.md) for what these
models do and do not achieve.
