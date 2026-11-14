"""Deterministic synthetic task families and split generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .problem import Problem


@dataclass(frozen=True)
class TaskRecord:
    family: str
    instance: int
    split: str
    problem: Problem


def quadratic_task(
    key: Any, dimension: int, condition: float = 10.0, rotated: bool = False
) -> Problem:
    k1, k2, k3 = jax.random.split(key, 3)
    center = jax.random.uniform(k1, (dimension,), minval=-2, maxval=2)
    weights = jnp.geomspace(1.0, condition, dimension)
    perm = jax.random.permutation(k2, dimension)
    weights = weights[perm]
    if rotated:
        q, _ = jnp.linalg.qr(jax.random.normal(k3, (dimension, dimension)))
    else:
        q = jnp.eye(dimension)
    matrix = q.T @ jnp.diag(weights) @ q
    return Problem(
        lambda x: (x - center) @ matrix @ (x - center),
        (-5.0, 5.0),
        dimension,
        identifier="rotated_quadratic" if rotated else "quadratic",
        metadata={"known_optimum": 0.0, "center": center, "condition": condition},
    )


def analytic_task(family: str, dimension: int, seed: int = 0) -> Problem:
    """Construct a deterministic bounded nonconvex benchmark task."""
    key = jax.random.PRNGKey(seed)
    shift = jax.random.uniform(key, (dimension,), minval=-0.5, maxval=0.5)
    if family == "rosenbrock":

        def objective(x: Any) -> Any:
            return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        return Problem(
            objective,
            (-2.0, 2.0),
            dimension,
            identifier=family,
            metadata={"known_optimum": 0.0},
        )
    if family == "rastrigin":

        def objective(x: Any) -> Any:
            z = x - shift
            return 10.0 * dimension + jnp.sum(z**2 - 10.0 * jnp.cos(2 * jnp.pi * z))
    elif family == "ackley":

        def objective(x: Any) -> Any:
            z = x - shift
            return (
                -20 * jnp.exp(-0.2 * jnp.sqrt(jnp.mean(z**2)))
                - jnp.exp(jnp.mean(jnp.cos(2 * jnp.pi * z)))
                + 20
                + jnp.e
            )
    else:
        raise ValueError(f"unsupported family: {family}")
    return Problem(
        objective,
        (-5.0, 5.0),
        dimension,
        identifier=family,
        metadata={"known_optimum": 0.0, "center": shift},
    )


def task_split(seed: int, count: int, dimension: int) -> list[TaskRecord]:
    keys = jax.random.split(jax.random.PRNGKey(seed), count)
    cuts = (int(0.7 * count), int(0.85 * count))
    return [
        TaskRecord(
            "quadratic",
            i,
            "train" if i < cuts[0] else "validation" if i < cuts[1] else "test",
            quadratic_task(k, dimension, 10.0 ** (i % 4)),
        )
        for i, k in enumerate(keys)
    ]
