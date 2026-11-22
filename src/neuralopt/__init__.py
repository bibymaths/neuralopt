"""Public API for neuralopt."""

from importlib.metadata import PackageNotFoundError, version

from .certification import Certificate, Expr, Variable, verify_certificate
from .hamiltonian import HamiltonianConfig
from .langevin import OverdampedConfig, UnderdampedConfig
from .model import ModelConfig, init_model
from .optimizer import OptimizationResult, minimize
from .problem import Problem
from .training import StochasticTrainingConfig

__all__ = [
    "Certificate",
    "Expr",
    "HamiltonianConfig",
    "ModelConfig",
    "OptimizationResult",
    "OverdampedConfig",
    "Problem",
    "StochasticTrainingConfig",
    "Variable",
    "UnderdampedConfig",
    "init_model",
    "minimize",
    "verify_certificate",
]
try:
    __version__ = version("neuralopt")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "0+unknown"
