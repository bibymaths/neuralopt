"""Outward-rounded interval expressions and branch-and-bound certification.

The backend uses NumPy ``nextafter`` after every elementary operation. This is
sound for the documented IEEE-754 operations when NumPy evaluates them in the
declared binary64 precision; it is intentionally separate from JAX/XLA.
"""

from __future__ import annotations

import heapq
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


def _down(x: Any) -> Any:
    return np.nextafter(x, -np.inf)


def _up(x: Any) -> Any:
    return np.nextafter(x, np.inf)


@dataclass(frozen=True)
class Interval:
    lo: Any
    hi: Any

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.lo) > np.asarray(self.hi)):
            raise ValueError("invalid interval")

    def __add__(self, other: Any) -> Interval:
        o = as_interval(other)
        return Interval(_down(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self) -> Interval:
        return Interval(_down(-self.hi), _up(-self.lo))

    def __sub__(self, other: Any) -> Interval:
        return self + -as_interval(other)

    def __rsub__(self, other: Any) -> Interval:
        return as_interval(other) - self

    def __mul__(self, other: Any) -> Interval:
        o = as_interval(other)
        q = np.stack((self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi))
        return Interval(_down(np.min(q, axis=0)), _up(np.max(q, axis=0)))

    __rmul__ = __mul__

    def square(self) -> Interval:
        lo = np.where((self.lo <= 0) & (self.hi >= 0), 0.0, np.minimum(self.lo**2, self.hi**2))
        return Interval(_down(lo), _up(np.maximum(self.lo**2, self.hi**2)))

    def exp(self) -> Interval:
        return Interval(_down(np.exp(self.lo)), _up(np.exp(self.hi)))

    def sin(self) -> Interval:
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        width = hi - lo
        s_lo, s_hi = np.minimum(np.sin(lo), np.sin(hi)), np.maximum(np.sin(lo), np.sin(hi))
        kmax = np.ceil((lo - np.pi / 2) / (2 * np.pi))
        kmin = np.ceil((lo + np.pi / 2) / (2 * np.pi))
        s_hi = np.where((width >= 2 * np.pi) | (np.pi / 2 + 2 * np.pi * kmax <= hi), 1.0, s_hi)
        s_lo = np.where((width >= 2 * np.pi) | (-np.pi / 2 + 2 * np.pi * kmin <= hi), -1.0, s_lo)
        return Interval(_down(s_lo), _up(s_hi))


def as_interval(x: Any) -> Interval:
    return (
        x
        if isinstance(x, Interval)
        else Interval(np.asarray(x, dtype=np.float64), np.asarray(x, dtype=np.float64))
    )


@dataclass(frozen=True)
class Expr:
    """Serializable expression over a bounded vector variable."""

    op: str
    args: tuple[Any, ...]

    def _binary(self, op: str, other: Any) -> Expr:
        return Expr(op, (self, to_expr(other)))

    def __add__(self, o: Any) -> Expr:
        return self._binary("add", o)

    __radd__ = __add__

    def __sub__(self, o: Any) -> Expr:
        return self._binary("sub", o)

    def __rsub__(self, o: Any) -> Expr:
        return to_expr(o)._binary("sub", self)

    def __mul__(self, o: Any) -> Expr:
        return self._binary("mul", o)

    __rmul__ = __mul__

    def __pow__(self, power: int) -> Expr:
        if power != 2:
            raise ValueError("certified expressions only support power 2")
        return Expr("square", (self,))

    def sin(self) -> Expr:
        return Expr("sin", (self,))

    def exp(self) -> Expr:
        return Expr("exp", (self,))

    def evaluate(self, x: Any) -> float:
        return float(_eval(self, np.asarray(x, dtype=np.float64), False))

    def interval(self, lo: Any, hi: Any) -> Interval:
        return _eval(self, Interval(np.asarray(lo, float), np.asarray(hi, float)), True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": [a.to_dict() if isinstance(a, Expr) else a for a in self.args],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Expr:
        return Expr(
            d["op"],
            tuple(Expr.from_dict(a) if isinstance(a, dict) and "op" in a else a for a in d["args"]),
        )


class Variable:
    """Factory for indexed expression variables."""

    def __getitem__(self, i: int) -> Expr:
        return Expr("var", (int(i),))


def to_expr(x: Any) -> Expr:
    return x if isinstance(x, Expr) else Expr("const", (float(x),))


def _eval(e: Expr, x: Any, interval: bool) -> Any:
    if e.op == "const":
        return as_interval(e.args[0]) if interval else e.args[0]
    if e.op == "var":
        return Interval(x.lo[e.args[0]], x.hi[e.args[0]]) if interval else x[e.args[0]]
    a = _eval(e.args[0], x, interval)
    if e.op in {"square", "sin", "exp"}:
        return (
            getattr(a, e.op)()
            if interval
            else getattr(np, e.op if e.op != "square" else "square")(a)
        )
    b = _eval(e.args[1], x, interval)
    return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[e.op]()


@dataclass(frozen=True)
class Certificate:
    problem_id: str
    expression: dict[str, Any]
    domain_lower: list[float]
    domain_upper: list[float]
    incumbent_x: list[float]
    upper_bound: float
    lower_bound: float
    absolute_gap: float
    relative_gap: float | None
    active_regions: list[dict[str, Any]]
    nodes_processed: int
    regions_pruned: int
    arithmetic: str
    precision: str
    status: str
    software_version: str = "0.2.0"
    environment: str = platform.platform()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @staticmethod
    def from_json(value: str) -> Certificate:
        return Certificate(**json.loads(value))


def verify_certificate(cert: Certificate, tolerance: float = 0.0) -> bool:
    """Replay leaf bounds and domain coverage independently of the model."""
    expr = Expr.from_dict(cert.expression)
    if not cert.active_regions:
        return False
    lbs = []
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    domain_lo = np.asarray(cert.domain_lower, dtype=float)
    domain_hi = np.asarray(cert.domain_upper, dtype=float)
    for r in cert.active_regions:
        lo = np.asarray(r["lo"], dtype=float)
        hi = np.asarray(r["hi"], dtype=float)
        if np.any(lo < domain_lo) or np.any(hi > domain_hi) or np.any(lo >= hi):
            return False
        boxes.append((lo, hi))
        iv = expr.interval(lo, hi)
        lbs.append(float(iv.lo))
        if float(iv.lo) + tolerance < r["lower_bound"]:
            return False
    domain_volume = float(np.prod(domain_hi - domain_lo))
    leaf_volume = float(sum(np.prod(hi - lo) for lo, hi in boxes))
    if not np.isclose(leaf_volume, domain_volume, rtol=1e-12, atol=tolerance):
        return False
    if not _is_binary_partition(boxes, domain_lo, domain_hi, tolerance):
        return False
    value = expr.evaluate(cert.incumbent_x)
    return (
        value <= cert.upper_bound
        and min(lbs) >= cert.lower_bound - tolerance
        and cert.upper_bound - min(lbs) <= cert.absolute_gap + tolerance
    )


def _is_binary_partition(
    boxes: list[tuple[np.ndarray, np.ndarray]],
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
    tolerance: float,
) -> bool:
    """Verify a guillotine box partition without all-pairs overlap tests.

    Branch-and-bound creates leaves by recursively bisecting a parent box. At
    each recursion this routine finds a shared endpoint crossed by no leaf and
    chooses the most balanced valid split. Valid replay is typically
    ``O(n log² n)`` rather than ``O(n²)`` in the number of leaves.
    """
    lower = np.stack([box[0] for box in boxes])
    upper = np.stack([box[1] for box in boxes])

    def visit(indices: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
        if indices.size == 1:
            i = int(indices[0])
            return bool(
                np.allclose(lower[i], lo, rtol=0.0, atol=tolerance)
                and np.allclose(upper[i], hi, rtol=0.0, atol=tolerance)
            )

        best: tuple[int, int, float, np.ndarray, np.ndarray] | None = None
        for dimension in range(lo.size):
            starts = np.sort(lower[indices, dimension])
            ends = np.sort(upper[indices, dimension])
            candidates = np.unique(np.concatenate((starts, ends)))
            candidates = candidates[(candidates > lo[dimension]) & (candidates < hi[dimension])]
            if not candidates.size:
                continue
            left_counts = np.searchsorted(ends, candidates, side="right")
            right_counts = indices.size - np.searchsorted(starts, candidates, side="left")
            valid = (left_counts > 0) & (right_counts > 0)
            valid &= left_counts + right_counts == indices.size
            if not np.any(valid):
                continue
            positions = np.flatnonzero(valid)
            position = int(
                positions[np.argmax(np.minimum(left_counts[valid], right_counts[valid]))]
            )
            split = float(candidates[position])
            left = indices[upper[indices, dimension] <= split]
            right = indices[lower[indices, dimension] >= split]
            balance = min(left.size, right.size)
            if best is None or balance > best[0]:
                best = (balance, dimension, split, left, right)
        if best is None:
            return False
        _, dimension, split, left, right = best
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[dimension] = split
        right_lo[dimension] = split
        return visit(left, lo, left_hi) and visit(right, right_lo, hi)

    return visit(np.arange(len(boxes)), domain_lo, domain_hi)


def certify(
    expr: Expr,
    lo: Any,
    hi: Any,
    gap: float,
    max_nodes: int,
    problem_id: str,
    priority: Literal["lower_bound", "widest"] = "lower_bound",
    split: Literal["widest", "round_robin"] = "widest",
    region_score: Callable[[np.ndarray, np.ndarray, float, float], float] | None = None,
    split_selector: Callable[[np.ndarray, np.ndarray], int] | None = None,
) -> Certificate:
    """Interval branch-and-bound with midpoint incumbents and widest splits."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    mid = (lo + hi) / 2
    best_x, upper = mid, _up(expr.evaluate(mid)).item()
    counter, processed, pruned = 0, 0, 0
    lower = float(expr.interval(lo, hi).lo)

    def score(a: np.ndarray, b: np.ndarray, lb: float) -> float:
        if region_score is not None:
            return float(region_score(a, b, lb, upper))
        return lb if priority == "lower_bound" else -float(np.max(b - a))

    heap: list[tuple[float, int, float, np.ndarray, np.ndarray]] = [
        (score(lo, hi, lower), counter, lower, lo, hi)
    ]
    separate_lower_bounds = region_score is not None or priority != "lower_bound"
    lower_heap: list[tuple[float, int]] = [(lower, counter)] if separate_lower_bounds else []
    active_ids = {counter} if separate_lower_bounds else set()
    terminal: list[tuple[float, np.ndarray, np.ndarray]] = []

    def active_lower_bound() -> float:
        if not separate_lower_bounds:
            return heap[0][2]
        while lower_heap and lower_heap[0][1] not in active_ids:
            heapq.heappop(lower_heap)
        return lower_heap[0][0]

    while heap and upper - active_lower_bound() > gap and processed < max_nodes:
        _, node_id, lb, a, b = heapq.heappop(heap)
        if separate_lower_bounds:
            active_ids.remove(node_id)
        processed += 1
        if lb >= upper:
            pruned += 1
            terminal.append((lb, a, b))
            continue
        if split_selector is not None:
            d = int(split_selector(a, b))
        elif split == "round_robin":
            d = (processed - 1) % len(a)
        else:
            d = int(np.argmax(b - a))
        if d < 0 or d >= len(a) or not b[d] > a[d]:
            raise ValueError("split policy selected an invalid dimension")
        m = (a[d] + b[d]) / 2
        for aa, bb in ((a.copy(), b.copy()), (a.copy(), b.copy())):
            if aa[d] == a[d]:
                bb[d] = m
            else:
                aa[d] = m
        # Explicit children avoids alias-sensitive branching.
        children = [(a.copy(), b.copy()), (a.copy(), b.copy())]
        children[0][1][d], children[1][0][d] = m, m
        for ca, cb in children:
            cm = (ca + cb) / 2
            value = _up(expr.evaluate(cm)).item()
            if value < upper:
                upper, best_x = value, cm
            cl = float(expr.interval(ca, cb).lo)
            if cl < upper:
                counter += 1
                heapq.heappush(heap, (score(ca, cb, cl), counter, cl, ca, cb))
                if separate_lower_bounds:
                    active_ids.add(counter)
                    heapq.heappush(lower_heap, (cl, counter))
            else:
                pruned += 1
                terminal.append((cl, ca, cb))
    retained = [(lb, a, b) for _, _, lb, a, b in heap] + terminal
    leaves = [{"lo": a.tolist(), "hi": b.tolist(), "lower_bound": lb} for lb, a, b in retained]
    global_lb = min((r["lower_bound"] for r in leaves), default=upper)
    status = "certified_gap" if upper - global_lb <= gap else "budget_exhausted"
    rel = (upper - global_lb) / max(1.0, abs(upper))
    return Certificate(
        problem_id,
        expr.to_dict(),
        lo.tolist(),
        hi.tolist(),
        best_x.tolist(),
        float(upper),
        float(global_lb),
        float(gap),
        float(rel),
        leaves,
        processed,
        pruned,
        "numpy-nextafter",
        "IEEE-754 binary64",
        status,
    )
