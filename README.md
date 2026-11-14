# neuralopt

`neuralopt` is an alpha research package for testing learned numerical optimization and restricted, replayable global certificates. A compact transformer consumes normalized parameter, incumbent, gradient, bounds, and budget features and directly emits the next optimization actions. No conventional optimizer runs during inference.

```python
import jax.numpy as jnp
from neuralopt import Problem, minimize

problem = Problem(lambda x: jnp.sum((x - 2) ** 2), (-5., 5.), 10)
result = minimize(problem, max_evaluations=100, seed=42)
```

The default API loads the bundled `demo-v0.1.0` checkpoint, trained for 100 updates on deterministic synthetic quadratics. It is a smoke artifact, not a generally validated optimizer. Pass explicit parameters to run another checkpoint or an ablation.

Learned mode requires differentiability or an explicit gradient. The included demo training command produces actual trained weights:

```bash
uv run neuralopt train-demo --steps 100 --output model.npz
```

Certified mode accepts only the package's serializable expression subset (`+`, `-`, `*`, square, `sin`, and `exp`). It uses binary64 outward rounding via `numpy.nextafter`, interval branch-and-bound, and a model-independent leaf replay verifier. It does not certify arbitrary JAX code.

```python
from neuralopt import Problem, Variable, minimize, verify_certificate
x = Variable()
expr = (x[0] - 0.25) ** 2
p = Problem(lambda z: (z[0] - .25) ** 2, (-1., 1.), 1, certified_expression=expr)
r = minimize(p, mode="certified", absolute_gap=1e-4, max_evaluations=10000)
assert verify_certificate(r.certificate)
```

See [limitations](docs/research/limitations.md) before interpreting experimental or certification results.
