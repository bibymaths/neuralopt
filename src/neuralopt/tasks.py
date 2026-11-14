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


def quadratic_task(key: Any, dimension: int, condition: float = 10.0) -> Problem:
    k1, k2 = jax.random.split(key)
    center = jax.random.uniform(k1, (dimension,), minval=-2, maxval=2)
    weights = jnp.geomspace(1.0, condition, dimension)
    perm = jax.random.permutation(k2, dimension)
    weights = weights[perm]
    return Problem(
        lambda x: jnp.sum(weights * (x - center) ** 2),
        (-5.0, 5.0),
        dimension,
        identifier="quadratic",
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
