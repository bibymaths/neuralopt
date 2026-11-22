# Quickstart

## Learned mode

```python
import jax.numpy as jnp
from neuralopt import Problem, minimize

problem = Problem(lambda x: jnp.sum((x - 2) ** 2), (-5.0, 5.0), 10)
result = minimize(problem, max_evaluations=100, seed=42)
print(result.status, result.fun, result.evaluations)
```

The default call loads the bundled `demo-v0.1.0` checkpoint, trained for 100 updates on
synthetic quadratics. It is a smoke artifact, not a validated general optimizer: on this
problem it stops at f ≈ 0.37 after 100 evaluations, where the minimum is 0. Pass
`params=` and `config=` to run another checkpoint.

## The preconditioned loop (compiled)

```python
result = minimize(problem, max_evaluations=100, candidate_method="preconditioned",
                  record_diagnostics=True)
d = result.diagnostics
print(d["engine"], result.fun, d["objective_evaluations"], d["gradient_evaluations"])
```

This runs the v0.6 step-acceptance loop as one compiled XLA program (the default engine
since v0.7.0). See [Compiled engine](guide/compiled-engine.md).

## Certified mode

```python
from neuralopt import Problem, Variable, minimize, verify_certificate

x = Variable()
expr = (x[0] - 0.25) ** 2
p = Problem(lambda z: (z[0] - 0.25) ** 2, (-1.0, 1.0), 1, certified_expression=expr)
r = minimize(p, mode="certified", absolute_gap=1e-4, max_evaluations=10000)
assert verify_certificate(r.certificate)
print(r.status, r.lower_bound, r.upper_bound)
```

Certified mode accepts only the package's expression subset. See
[Certified mode](guide/certified-mode.md).

## Train a demo checkpoint

```bash
pip install "neuralopt[train]"
neuralopt train-demo --steps 100 --output model.npz
```
