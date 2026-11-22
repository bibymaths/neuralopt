# Installation

NeuralOpt needs Python 3.11 or newer. It depends on JAX and NumPy only.

```bash
pip install neuralopt
```

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `neuralopt[train]` | Optax | training checkpoints (`neuralopt train-demo`, `train_batched`) |
| `neuralopt[benchmark]` | SciPy, Optimistix | the benchmark scripts' reference solvers (never used in inference) |
| `neuralopt[sde]` | Diffrax | the optional SDE reference checks |

JAX installs its CPU build by default. For GPU support, follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## From source

```bash
git clone https://github.com/bibymaths/neuralopt
cd neuralopt
uv sync --extra dev            # tests, lint, type checks
uv sync --extra dev --group docs   # plus the documentation toolchain
uv run pytest -q
```

Versions before 0.7.0 are not on PyPI. To use one, install from its git tag, for
example `pip install "git+https://github.com/bibymaths/neuralopt@v0.6.0"`.
