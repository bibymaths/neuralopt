<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bibymaths/neuralopt/main/docs/assets/images/logo-dark.png">
    <img src="https://raw.githubusercontent.com/bibymaths/neuralopt/main/docs/assets/images/logo.png" alt="NeuralOpt" width="420">
  </picture>
</p>

<p align="center"><b>A JAX research package for learned optimization, with replayable interval certification.</b></p>

<p align="center">
  <a href="https://github.com/bibymaths/neuralopt/actions/workflows/ci.yml"><img src="https://github.com/bibymaths/neuralopt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/bibymaths/neuralopt/actions/workflows/docs.yml"><img src="https://github.com/bibymaths/neuralopt/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://pypi.org/project/neuralopt/"><img src="https://img.shields.io/pypi/v/neuralopt" alt="PyPI version"></a>
  <a href="https://pypi.org/project/neuralopt/"><img src="https://img.shields.io/pypi/pyversions/neuralopt" alt="Python versions"></a>
  <a href="https://github.com/bibymaths/neuralopt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
</p>

> **Status: alpha research package.** In our own studies (v0.4–v0.7), the learned
> geometries did not beat validation-tuned fixed or Barzilai–Borwein baselines, and
> classical L-BFGS outperforms the learned optimizer on the reported benchmarks. To solve
> a problem, use SciPy or Optimistix; see [when to use NeuralOpt](https://bibymaths.github.io/neuralopt/latest/when-to-use/).

NeuralOpt has two independent modes:

- **Learned mode:** a compact transformer reads normalized position, gradient, bounds
  and budget features (and, opt-in, a masked history of evaluated states) and proposes
  the next step. No conventional optimizer runs during inference, and every objective
  and gradient evaluation is counted against one budget.
- **Certified mode:** interval branch-and-bound with outward rounding proves global
  bounds for a restricted expression language, with no learned component. Certificates
  can be replayed by an independent verifier.

## Install

```bash
pip install neuralopt
```

Python 3.11+. Extras: `neuralopt[train]` (Optax), `neuralopt[benchmark]` (SciPy,
Optimistix), `neuralopt[sde]` (Diffrax). For development: `uv sync --extra dev`.

## Quickstart

```python
import jax.numpy as jnp
from neuralopt import Problem, minimize

problem = Problem(lambda x: jnp.sum((x - 2) ** 2), (-5.0, 5.0), 10)
result = minimize(problem, max_evaluations=100, seed=42)
```

The default call loads the bundled `demo-v0.1.0` checkpoint, a smoke artifact trained
for 100 updates on synthetic quadratics; it is not a validated general optimizer. The
v0.6 preconditioned loop runs compiled by default (`candidate_method="preconditioned"`).

Certified mode accepts only the package's expression subset (`+`, `-`, `*`, square,
`sin`, `exp`). It uses binary64 outward rounding via `numpy.nextafter` and an
independent leaf-replay verifier; it does not certify arbitrary JAX code.

```python
from neuralopt import Problem, Variable, minimize, verify_certificate

x = Variable()
expr = (x[0] - 0.25) ** 2
p = Problem(lambda z: (z[0] - 0.25) ** 2, (-1.0, 1.0), 1, certified_expression=expr)
r = minimize(p, mode="certified", absolute_gap=1e-4, max_evaluations=10000)
assert verify_certificate(r.certificate)
```

## When to use NeuralOpt

- **To solve an optimization problem:** use SciPy L-BFGS-B or Optimistix. In the v0.7
  equal-work benchmark they reached 1e-6 regret on 32/32 tasks; NeuralOpt's learned
  geometry reached it on 0/32.
- **To research learned optimizers** with honest accounting (shared evaluation budget,
  pre-registered studies, versioned checkpoints): use NeuralOpt.
- **For a replayable global certificate** of a small expression in the supported subset:
  use certified mode.

## Documentation

- Docs: <https://bibymaths.github.io/neuralopt/>
- [Release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/) and
  [CHANGELOG](https://github.com/bibymaths/neuralopt/blob/main/CHANGELOG.md)
- [Limitations](https://bibymaths.github.io/neuralopt/latest/limitations/), to read
  before interpreting any result
- [Contributing](https://github.com/bibymaths/neuralopt/blob/main/CONTRIBUTING.md)

## Citation

Please cite the exact version or commit you used; see
[Citing NeuralOpt](https://bibymaths.github.io/neuralopt/latest/citing/) and
[`CITATION.cff`](https://github.com/bibymaths/neuralopt/blob/main/CITATION.cff).

```bibtex
@software{mishra_neuralopt,
  author  = {Mishra, Abhinav},
  title   = {{NeuralOpt: a JAX research package for learned optimization, with replayable interval certification}},
  version = {0.7.0},
  year    = {2026},
  url     = {https://github.com/bibymaths/neuralopt}
}
```

## License

MIT; see [LICENSE](https://github.com/bibymaths/neuralopt/blob/main/LICENSE).
