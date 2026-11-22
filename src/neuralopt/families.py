"""v0.8 problem families ``f(θ; c)`` with fixed, hashed splits.

See ``docs/plans/neuralopt-v08-amortized-families.md`` (sections 7-8). Every family has a
module-level objective ``f(x, c)`` (static for ``jit``/``vmap``); ``x`` is the parameter
vector θ and ``c`` is the instance context: padded observation arrays plus a dose.

Context layout (one instance; ``n = max_observations``):
``t`` (n,) observation times · ``t_idx`` (n,) int index on the solver grid (ODE families)
· ``y`` (n,) observed values · ``species`` (n,) int channel id · ``sigma`` (n,) noise
standard deviation · ``mask`` (n,) bool · ``dose`` () condition value.

Objective: weighted least squares ``0.5 · mean_mask(((model − y) / sigma)²)``, about 0.5 at
the truth. Instance ``i`` of split ``s`` draws all its randomness from
``sha256(f"{family}|{s}|{i}")``, so it does not depend on how many instances are requested.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

IN_DISTRIBUTION = ("train", "validation", "test")


def _rng(family: str, split: str, i: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{family}|{split}|{i}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _weighted_lsq(model: Any, c: dict[str, Any]) -> Any:
    r = jnp.where(c["mask"], (model - c["y"]) / c["sigma"], 0.0)
    return 0.5 * jnp.sum(r * r) / jnp.sum(c["mask"])


# --- family 1: damped sinusoid --------------------------------------------------------


def sinusoid_observe(x: Any, c: dict[str, Any], refine: int = 1) -> Any:
    a, lam, omega, phi = x[0], x[1], x[2], x[3]
    return a * jnp.exp(-lam * c["t"]) * jnp.sin(2 * jnp.pi * omega * c["t"] + phi)


def sinusoid_objective(x: Any, c: dict[str, Any]) -> Any:
    return _weighted_lsq(sinusoid_observe(x, c), c)


# --- family 2: two-step Michaelis-Menten cascade S -(E1)-> I -(E2)-> P -----------------

CASCADE_DT, CASCADE_STEPS = 0.01, 1000  # t in [0, 10]


def _cascade_rhs(z: Any, p: Any) -> Any:
    v1, k1, v2, k2 = p[0], p[1], p[2], p[3]
    s, i = z[0], z[1]
    r1, r2 = v1 * s / (k1 + s), v2 * i / (k2 + i)
    return jnp.stack([-r1, r1 - r2, r2])


def cascade_trajectory(x: Any, dose: Any, refine: int = 1) -> Any:
    """States (S, I, P) at the base grid ``k · CASCADE_DT``, k = 0..CASCADE_STEPS (RK4)."""
    p = jnp.exp(x)
    dt = CASCADE_DT / refine
    z0 = jnp.stack([dose, jnp.zeros_like(dose), jnp.zeros_like(dose)])

    def rk4(z: Any, _: Any) -> tuple[Any, None]:
        k1 = _cascade_rhs(z, p)
        k2 = _cascade_rhs(z + 0.5 * dt * k1, p)
        k3 = _cascade_rhs(z + 0.5 * dt * k2, p)
        k4 = _cascade_rhs(z + dt * k3, p)
        return z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, None

    def base_step(z: Any, _: Any) -> tuple[Any, Any]:
        z, _ = jax.lax.scan(rk4, z, None, length=refine)
        return z, z

    _, zs = jax.lax.scan(base_step, z0, None, length=CASCADE_STEPS)
    return jnp.concatenate([z0[None], zs])


def cascade_observe(x: Any, c: dict[str, Any], refine: int = 1) -> Any:
    states = cascade_trajectory(x, c["dose"], refine)
    return states[c["t_idx"], c["species"] + 1]  # species 0 -> I, 1 -> P


def cascade_objective(x: Any, c: dict[str, Any]) -> Any:
    return _weighted_lsq(cascade_observe(x, c), c)


# --- specs and split generation ---------------------------------------------------------


@dataclass(frozen=True)
class FamilySpec:
    name: str
    dimension: int
    lo: Any
    hi: Any
    prior_lo: Any
    prior_hi: Any
    default: Any  # nominal start (prior centre)
    max_observations: int
    splits: tuple[str, ...]
    objective: Callable[[Any, Any], Any]
    observe: Callable[..., Any]
    design: Callable[[np.random.Generator, str], dict[str, Any]]  # truth, design, noise draws
    solver_tolerance: float = 0.0  # max relative RK4 error versus a 16x finer step


def _pad(values: list[Any], n: int, fill: float = 0.0, dtype: Any = float) -> np.ndarray:
    out = np.full(n, fill, dtype=dtype)
    out[: len(values)] = values
    return out


def _sinusoid_design(rng: np.random.Generator, split: str) -> dict[str, Any]:
    n = 40
    truth = np.array([rng.uniform(0.5, 2.0), rng.uniform(0.05, 0.5), rng.uniform(0.5, 2.5),
                      rng.uniform(-np.pi, np.pi)])
    if split == "ood-params":  # frequency above the training prior (aliasing), in the box
        truth[2] = rng.uniform(2.6, 3.5)
    if split == "ood-sampling":
        k = 15
        t = np.sort(rng.uniform(0.0, 10.0, k))
    else:
        k = n
        t = np.clip(0.25 * np.arange(n) + rng.uniform(-0.05, 0.05, n), 0.0, None)
    sigma = 0.3 if split == "ood-noise" else 0.1
    return {"truth": truth, "t": _pad(list(t), n), "t_idx": np.zeros(n, int),
            "species": np.zeros(n, int), "mask": _pad([True] * k, n, False, bool),
            "noise": rng.standard_normal(n), "sigma_rule": ("absolute", sigma), "dose": 0.0}


def _grid_index(t: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(t) / CASCADE_DT).astype(int)


def _cascade_truth(rng: np.random.Generator, split: str) -> np.ndarray:
    logv = rng.uniform(np.log(0.2), np.log(2.0), 2)
    logk = rng.uniform(np.log(0.2), np.log(5.0), 2)
    if split == "ood-params":  # V1 above the training prior, inside the box
        logv[0] = rng.uniform(np.log(2.5), np.log(6.0))
    return np.array([logv[0], logk[0], logv[1], logk[1]])


def _cascade_clean_design(rng: np.random.Generator, split: str) -> dict[str, Any]:
    n = 40
    truth = _cascade_truth(rng, split)
    dose = rng.uniform(6.0, 10.0) if split == "ood-dose" else rng.uniform(1.0, 3.0)
    if split == "ood-sampling":
        times = 0.5 * np.arange(1, 7)  # six early times only
    else:
        times = 0.5 * np.arange(1, 21)
    t = np.concatenate([times, times])
    species = np.concatenate([np.zeros(len(times), int), np.ones(len(times), int)])
    sigma = 0.02 if split == "ood-noise" else 0.01
    k = len(t)
    return {"truth": truth, "t": _pad(list(t), n), "t_idx": _pad(list(_grid_index(t)), n, 0, int),
            "species": _pad(list(species), n, 0, int), "mask": _pad([True] * k, n, False, bool),
            "noise": rng.standard_normal(n), "sigma_rule": ("absolute", sigma), "dose": dose}


def _cascade_realistic_design(rng: np.random.Generator, split: str) -> dict[str, Any]:
    n = 16
    truth = _cascade_truth(rng, split)
    dose = rng.uniform(6.0, 10.0) if split == "ood-dose" else rng.uniform(0.5, 5.0)
    if split == "ood-sampling":
        grid = 0.2 * np.arange(1, 16)  # early times only, [0.2, 3]
        k = int(rng.integers(3, 5))
    else:
        grid = 0.2 * np.arange(1, 51)  # [0.2, 10]
        k = int(rng.integers(5, 9))
    times = np.sort(rng.choice(grid, size=k, replace=False))
    i_times = np.sort(rng.choice(times, size=max(1, k // 2), replace=False))
    t = np.concatenate([times, i_times])
    species = np.concatenate([np.ones(k, int), np.zeros(len(i_times), int)])  # P all, I subset
    level = rng.uniform(0.05, 0.10) * (2.0 if split == "ood-noise" else 1.0)
    m = len(t)
    return {"truth": truth, "t": _pad(list(t), n), "t_idx": _pad(list(_grid_index(t)), n, 0, int),
            "species": _pad(list(species), n, 0, int), "mask": _pad([True] * m, n, False, bool),
            "noise": rng.standard_normal(n), "sigma_rule": ("relative", level), "dose": dose}


def _box(lo: list[float], hi: list[float]) -> tuple[Any, Any]:
    return np.asarray(lo, float), np.asarray(hi, float)


_SPLITS = (*IN_DISTRIBUTION, "ood-noise", "ood-params", "ood-sampling")
_CASCADE_LO, _CASCADE_HI = _box(list(np.log([0.02, 0.05, 0.02, 0.05])),
                                list(np.log([10.0, 20.0, 10.0, 20.0])))
_CASCADE_PRIOR = _box(list(np.log([0.2, 0.2, 0.2, 0.2])), list(np.log([2.0, 5.0, 2.0, 5.0])))

FAMILIES: dict[str, FamilySpec] = {
    "sinusoid": FamilySpec(
        name="sinusoid", dimension=4,
        lo=np.array([0.1, 0.0, 0.1, -np.pi]), hi=np.array([5.0, 2.0, 5.0, np.pi]),
        prior_lo=np.array([0.5, 0.05, 0.5, -np.pi]), prior_hi=np.array([2.0, 0.5, 2.5, np.pi]),
        default=np.array([1.25, 0.275, 1.5, 0.0]), max_observations=40, splits=_SPLITS,
        objective=sinusoid_objective, observe=sinusoid_observe, design=_sinusoid_design,
    ),
    "cascade-clean": FamilySpec(
        name="cascade-clean", dimension=4, lo=_CASCADE_LO, hi=_CASCADE_HI,
        prior_lo=_CASCADE_PRIOR[0], prior_hi=_CASCADE_PRIOR[1],
        default=0.5 * (_CASCADE_PRIOR[0] + _CASCADE_PRIOR[1]), max_observations=40,
        splits=(*_SPLITS, "ood-dose"), objective=cascade_objective, observe=cascade_observe,
        design=_cascade_clean_design, solver_tolerance=1e-8,
    ),
    "cascade-realistic": FamilySpec(
        name="cascade-realistic", dimension=4, lo=_CASCADE_LO, hi=_CASCADE_HI,
        prior_lo=_CASCADE_PRIOR[0], prior_hi=_CASCADE_PRIOR[1],
        default=0.5 * (_CASCADE_PRIOR[0] + _CASCADE_PRIOR[1]), max_observations=16,
        splits=(*_SPLITS, "ood-dose"), objective=cascade_objective, observe=cascade_observe,
        design=_cascade_realistic_design, solver_tolerance=1e-8,
    ),
}


def family_objective(name: str) -> Callable[[Any, Any], Any]:
    """The family's module-level objective ``f(x, c)`` (stable object: one compilation)."""
    if name in SYNTHETIC:
        return SYNTHETIC[name].objective
    return FAMILIES[name].objective


@partial(jax.jit, static_argnames=("name",))
def _observe_batch(name: str, truths: Any, contexts: Any) -> Any:
    return jax.vmap(FAMILIES[name].observe)(truths, contexts)


def make_split(name: str, split: str, n: int) -> tuple[dict[str, Any], Any]:
    """``n`` instances of ``split``: stacked contexts (leading axis) and true parameters.

    For the synthetic (v0.8-B) families the "true parameters" are the global minimizers.
    """
    if name in SYNTHETIC:
        return _synthetic_split(SYNTHETIC[name], split, n)
    spec = FAMILIES[name]
    if split not in spec.splits:
        raise ValueError(f"{name} has no split {split!r}; choose from {spec.splits}")
    designs = [spec.design(_rng(name, split, i), split) for i in range(n)]
    keys = ("t", "t_idx", "species", "mask")
    contexts = {k: jnp.asarray(np.stack([d[k] for d in designs])) for k in keys}
    contexts["dose"] = jnp.asarray(np.array([d["dose"] for d in designs], float))
    contexts["y"] = jnp.zeros_like(contexts["t"])
    contexts["sigma"] = jnp.ones_like(contexts["t"])
    truths = jnp.asarray(np.stack([d["truth"] for d in designs]))
    clean = np.asarray(_observe_batch(name, truths, contexts))
    sigma = np.empty_like(clean)
    for j, d in enumerate(designs):
        rule, level = d["sigma_rule"]
        if rule == "absolute":
            sigma[j] = level
        else:  # relative to the signal, floored at 5% of the dose
            sigma[j] = level * np.maximum(np.abs(clean[j]), 0.05 * d["dose"])
    noise = np.stack([d["noise"] for d in designs])
    mask = np.asarray(contexts["mask"])
    contexts["sigma"] = jnp.asarray(np.where(mask, sigma, 1.0))
    contexts["y"] = jnp.asarray(np.where(mask, clean + sigma * noise, 0.0))
    return contexts, truths


def split_digest(contexts: dict[str, Any], truths: Any) -> str:
    """sha256 over every context array (sorted keys) and the truths, as float64/int64 bytes."""
    h = hashlib.sha256()
    for key in sorted(contexts):
        a = np.asarray(contexts[key])
        h.update(key.encode())
        h.update(a.astype(np.int64 if a.dtype.kind in "biu" else np.float64).tobytes())
    h.update(np.asarray(truths, np.float64).tobytes())
    return h.hexdigest()


# --- v0.8-B synthetic families (docs/plans/neuralopt-v08b-step-policy.md, section 2) -----
#
# Context: ``shift`` o (d,) = the global minimizer, ``rotation`` R (d, d), ``amplitude`` A
# (Rastrigin only; 1 otherwise) and ``x0`` (d,), the shared first evaluation point drawn
# uniformly in the box. Global minimum f* = 0 at x = o for every instance and split. The
# policy must never read ``shift`` or ``rotation`` (they reveal the optimum).


def rastrigin_rotated_objective(x: Any, c: dict[str, Any]) -> Any:
    z = c["rotation"] @ (x - c["shift"])
    a = c["amplitude"]
    return a * z.shape[-1] + jnp.sum(z * z - a * jnp.cos(2 * jnp.pi * z))


def ackley_rotated_objective(x: Any, c: dict[str, Any]) -> Any:
    z = c["rotation"] @ (x - c["shift"])
    return (-20.0 * jnp.exp(-0.2 * jnp.sqrt(jnp.mean(z * z) + 1e-300))
            - jnp.exp(jnp.mean(jnp.cos(2 * jnp.pi * z))) + 20.0 + jnp.e)


def rosenbrock_shifted_objective(x: Any, c: dict[str, Any]) -> Any:
    z = x - c["shift"] + 1.0
    return jnp.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2)


@dataclass(frozen=True)
class SyntheticSpec:
    name: str
    dimension: int  # in distribution (ood-dimension uses 8)
    half_width: float  # box [-h, h]^d
    shift_half: float  # in-distribution shift o ~ U[-s, s]^d
    ood_shift: tuple[float, float]  # ood-shift: |o_j| ~ U[a, b], random sign
    rotated: bool
    amplitude: tuple[float, float] | None  # Rastrigin A range in distribution
    ood_amplitude: tuple[float, float] | None
    splits: tuple[str, ...]
    objective: Callable[[Any, Any], Any]

    def dimension_for(self, split: str) -> int:
        return 8 if split == "ood-dimension" else self.dimension

    def bounds(self, split: str = "validation") -> tuple[Any, Any]:
        d = self.dimension_for(split)
        return -self.half_width * np.ones(d), self.half_width * np.ones(d)


def _haar_rotation(rng: np.random.Generator, d: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def _synthetic_instance(spec: SyntheticSpec, rng: np.random.Generator,
                        split: str) -> dict[str, Any]:
    d = spec.dimension_for(split)
    if split == "ood-shift":
        a, b = spec.ood_shift
        shift = rng.uniform(a, b, d) * rng.choice([-1.0, 1.0], d)
    else:
        shift = rng.uniform(-spec.shift_half, spec.shift_half, d)
    rotation = _haar_rotation(rng, d) if spec.rotated else np.eye(d)
    if spec.amplitude is None:
        amplitude = 1.0
    else:
        lo, hi = spec.ood_amplitude if split == "ood-amplitude" and spec.ood_amplitude \
            else spec.amplitude
        amplitude = rng.uniform(lo, hi)
    x0 = rng.uniform(-spec.half_width, spec.half_width, d)
    return {"shift": shift, "rotation": rotation, "amplitude": amplitude, "x0": x0}


def _synthetic_split(spec: SyntheticSpec, split: str, n: int) -> tuple[dict[str, Any], Any]:
    if split not in spec.splits:
        raise ValueError(f"{spec.name} has no split {split!r}; choose from {spec.splits}")
    rows = [_synthetic_instance(spec, _rng(spec.name, split, i), split) for i in range(n)]
    contexts = {k: jnp.asarray(np.stack([np.asarray(r[k], float) for r in rows]))
                for k in ("shift", "rotation", "amplitude", "x0")}
    return contexts, contexts["shift"]


_SYNTHETIC_SPLITS = (*IN_DISTRIBUTION, "ood-shift", "ood-dimension")

SYNTHETIC: dict[str, SyntheticSpec] = {
    "rastrigin": SyntheticSpec(
        name="rastrigin", dimension=4, half_width=5.12, shift_half=2.5, ood_shift=(2.5, 4.0),
        rotated=True, amplitude=(5.0, 15.0), ood_amplitude=(15.0, 25.0),
        splits=(*_SYNTHETIC_SPLITS, "ood-amplitude"), objective=rastrigin_rotated_objective,
    ),
    "ackley": SyntheticSpec(
        name="ackley", dimension=4, half_width=32.768, shift_half=16.0, ood_shift=(16.0, 28.0),
        rotated=True, amplitude=None, ood_amplitude=None, splits=_SYNTHETIC_SPLITS,
        objective=ackley_rotated_objective,
    ),
    "rosenbrock-shifted": SyntheticSpec(
        name="rosenbrock-shifted", dimension=4, half_width=5.0, shift_half=2.5,
        ood_shift=(2.5, 4.0), rotated=False, amplitude=None, ood_amplitude=None,
        splits=_SYNTHETIC_SPLITS, objective=rosenbrock_shifted_objective,
    ),
}
