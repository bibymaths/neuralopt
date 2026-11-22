"""Parametric synthetic task families: one compilation per family and dimension.

Each family has a single module-level objective ``f(x, theta)``. Task identity lives in
``theta`` (arrays), so ``jit``/``vmap`` reuse one compiled program for every task of the
same family and dimension. :func:`make_task` reproduces the v0.6 harness generator
(``scaled_task``: ``a * base(x / c)`` on the box scaled by ``c``) exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


def quadratic_objective(x: Any, theta: dict[str, Any]) -> Any:
    z = x / theta["coordinate_scale"] - theta["center"]
    return theta["scale"] * (z @ theta["matrix"] @ z)


def rastrigin_objective(x: Any, theta: dict[str, Any]) -> Any:
    z = x / theta["coordinate_scale"] - theta["shift"]
    return theta["scale"] * (10.0 * z.shape[-1] + jnp.sum(z**2 - 10.0 * jnp.cos(2 * jnp.pi * z)))


def ackley_objective(x: Any, theta: dict[str, Any]) -> Any:
    z = x / theta["coordinate_scale"] - theta["shift"]
    value = (
        -20 * jnp.exp(-0.2 * jnp.sqrt(jnp.mean(z**2)))
        - jnp.exp(jnp.mean(jnp.cos(2 * jnp.pi * z)))
        + 20
        + jnp.e
    )
    return theta["scale"] * value


def rosenbrock_objective(x: Any, theta: dict[str, Any]) -> Any:
    y = x / theta["coordinate_scale"]
    return theta["scale"] * jnp.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1 - y[:-1]) ** 2)


_OBJECTIVES = {
    "quadratic": quadratic_objective,
    "rastrigin": rastrigin_objective,
    "ackley": ackley_objective,
    "rosenbrock": rosenbrock_objective,
}


def family_objective(family: str) -> Any:
    """The (stable, module-level) objective ``f(x, theta)`` of a family."""
    return _OBJECTIVES[family]


@dataclass(frozen=True)
class ParametricTask:
    family: str  # objective family: quadratic, rastrigin, ackley, rosenbrock
    label: str  # e.g. "rotated-100"
    dimension: int
    theta: dict[str, Any]
    lo: Any
    hi: Any
    x0: Any
    known_optimum: float


def make_task(
    family: str,
    condition: float,
    dimension: int,
    seed: int,
    objective_scale: float = 1.0,
    coordinate_scale: float = 1.0,
) -> ParametricTask:
    """Build a task with the same randomness and scaling as the v0.6 generator."""
    a = jnp.asarray(objective_scale, dtype=jnp.result_type(float))
    c = jnp.asarray(coordinate_scale, dtype=a.dtype)
    key = jax.random.PRNGKey(seed)
    if family in ("isotropic", "diagonal", "rotated"):
        k1, k2, k3 = jax.random.split(key, 3)
        center = jax.random.uniform(k1, (dimension,), minval=-2, maxval=2)
        weights = jnp.geomspace(1.0, condition, dimension)[jax.random.permutation(k2, dimension)]
        if family == "rotated":
            q, _ = jnp.linalg.qr(jax.random.normal(k3, (dimension, dimension)))
            matrix = q.T @ jnp.diag(weights) @ q
        else:
            matrix = jnp.diag(weights)
        theta = {"scale": a, "coordinate_scale": c, "center": center, "matrix": matrix}
        objective_family, half = "quadratic", 5.0
        label = f"{family}-{int(condition)}" if family != "isotropic" else family
    elif family in ("rastrigin", "ackley"):
        shift = jax.random.uniform(key, (dimension,), minval=-0.5, maxval=0.5)
        theta = {"scale": a, "coordinate_scale": c, "shift": shift}
        objective_family, half, label = family, 5.0, family
    elif family == "rosenbrock":
        theta = {"scale": a, "coordinate_scale": c}
        objective_family, half, label = family, 2.0, family
    else:
        raise ValueError(f"unsupported family: {family}")
    lo = -half * c * jnp.ones((dimension,), a.dtype)
    hi = half * c * jnp.ones((dimension,), a.dtype)
    return ParametricTask(objective_family, label, dimension, theta, lo, hi,
                          jnp.zeros((dimension,), a.dtype), 0.0)


def stack_tasks(tasks: list[ParametricTask]) -> list[Any]:
    """Group tasks by (objective family, dimension) into batched :class:`TaskGroup` s."""
    from .fast_training import TaskGroup

    groups: dict[tuple[str, int], list[ParametricTask]] = {}
    for task in tasks:
        groups.setdefault((task.family, task.dimension), []).append(task)
    return [TaskGroup.from_tasks(members) for members in groups.values()]
