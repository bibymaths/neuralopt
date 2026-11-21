"""Compiled, batched history training (Track A5).

One jitted ``train_step`` per (family objective, dimension, batch size) computes the
batched loss, its gradient and the Optax update together. A step whose loss or
meta-gradient is nonfinite is *skipped* (parameters and optimizer state unchanged) and
reported in :class:`TrainingReport`; it is never dropped silently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .model import DEFAULT_CONFIG, ModelConfig, Params, init_mlp_model, init_model
from .training import HistoryUnrollConfig, history_regret_from_rollout, history_unroll_core


@dataclass(frozen=True)
class TaskGroup:
    """Tasks sharing one objective function and dimension; arrays carry a task axis."""

    name: str
    objective: Any
    thetas: Any
    lo: Any
    hi: Any
    optimum: Any

    @property
    def size(self) -> int:
        return int(self.lo.shape[0])

    @classmethod
    def from_tasks(cls, tasks: list[Any]) -> TaskGroup:
        from .task_families import family_objective

        families = {(t.family, t.dimension) for t in tasks}
        if len(families) != 1:
            raise ValueError("a TaskGroup needs one family and one dimension")
        family, dimension = families.pop()
        return cls(
            name=f"{family}-d{dimension}",
            objective=family_objective(family),
            thetas=jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[t.theta for t in tasks]),
            lo=jnp.stack([t.lo for t in tasks]),
            hi=jnp.stack([t.hi for t in tasks]),
            optimum=jnp.asarray([t.known_optimum for t in tasks], dtype=tasks[0].lo.dtype),
        )


class TrainStep:
    """A compiled ``(params, opt_state, batch...) -> (params, opt_state, loss, skipped)``."""

    def __init__(self, fn: Any, optimizer: Any) -> None:
        self._fn = fn
        self._optimizer = optimizer

    def init(self, params: Params) -> Any:
        return self._optimizer.init(params)

    def __call__(self, params: Params, opt_state: Any, thetas: Any, lo: Any, hi: Any,
                 optimum: Any, keys: Any) -> tuple[Params, Any, Any, Any]:
        return self._fn(params, opt_state, thetas, lo, hi, optimum, keys)


@cache
def make_train_step(
    objective: Any,
    config: ModelConfig,
    unroll: HistoryUnrollConfig,
    model_kind: str,
    learning_rate: float,
) -> TrainStep:
    """Build (once per argument tuple) the jitted train step for a family objective."""
    import optax

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))

    def train_step(params: Params, opt_state: Any, thetas: Any, lo: Any, hi: Any,
                   optimum: Any, keys: Any) -> tuple[Params, Any, Any, Any]:
        def loss_fn(p: Params) -> Any:
            def one(theta: Any, lo_i: Any, hi_i: Any, optimum_i: Any, key: Any) -> Any:
                rolled = history_unroll_core(
                    p, lambda x: objective(x, theta), (lo_i, hi_i), key, config, unroll,
                    model_kind,
                )
                return history_regret_from_rollout(rolled, optimum_i, unroll)

            return jnp.mean(jax.vmap(one)(thetas, lo, hi, optimum, keys))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        finite = jnp.isfinite(loss) & jnp.all(
            jnp.asarray([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)])
        )
        safe_grads = jax.tree_util.tree_map(lambda g: jnp.where(finite, g, 0.0), grads)
        updates, new_state = optimizer.update(safe_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        def keep(new: Any, old: Any) -> Any:
            return jax.tree_util.tree_map(lambda n, o: jnp.where(finite, n, o), new, old)

        return keep(new_params, params), keep(new_state, opt_state), loss, ~finite

    return TrainStep(jax.jit(train_step), optimizer)


@dataclass
class TrainingReport:
    params: Params
    losses: list[float]
    skipped_steps: list[int] = field(default_factory=list)
    compiled_groups: int = 0
    seconds: float = 0.0

    @property
    def skipped_updates(self) -> int:
        return len(self.skipped_steps)


def train_batched(
    groups: list[TaskGroup],
    *,
    seed: int = 0,
    steps: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 8,
    config: ModelConfig = DEFAULT_CONFIG,
    unroll: HistoryUnrollConfig | None = None,
    model_kind: str = "transformer",
) -> TrainingReport:
    """Cycle over task groups; each update uses a random batch from one group.

    ``losses[i]`` is the batch loss at step ``i`` (it may be finite even when the step
    was skipped for a nonfinite gradient). ``skipped_steps`` lists every skipped update.
    """
    if not groups:
        raise ValueError("at least one task group is required")
    unroll = unroll if unroll is not None else HistoryUnrollConfig()
    if model_kind not in {"transformer", "mlp"}:
        raise ValueError("model_kind must be 'transformer' or 'mlp'")
    key = jax.random.PRNGKey(seed)
    params = init_model(key, config) if model_kind == "transformer" else init_mlp_model(key, config)
    step_fns = {
        g.name: make_train_step(g.objective, config, unroll, model_kind, learning_rate)
        for g in groups
    }
    opt_state = next(iter(step_fns.values())).init(params)
    losses: list[float] = []
    skipped: list[int] = []
    start = time.perf_counter()
    for step in range(steps):
        group = groups[step % len(groups)]
        key, batch_key, loss_key = jax.random.split(key, 3)
        size = min(batch_size, group.size)
        index = np.asarray(jax.random.choice(batch_key, group.size, (size,), replace=False))
        batch = jax.tree_util.tree_map(lambda a, i=index: a[i], group.thetas)
        params, opt_state, loss, was_skipped = step_fns[group.name](
            params, opt_state, batch, group.lo[index], group.hi[index], group.optimum[index],
            jax.random.split(loss_key, size),
        )
        losses.append(float(loss))
        if bool(was_skipped):
            skipped.append(step)
    return TrainingReport(params, losses, skipped, len(step_fns), time.perf_counter() - start)
