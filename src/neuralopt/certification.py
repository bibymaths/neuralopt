"""Outward-rounded interval expressions and branch-and-bound certification.

The backend uses NumPy ``nextafter`` after every elementary operation. This is
sound for the documented IEEE-754 operations when NumPy evaluates them in the
declared binary64 precision; it is intentionally separate from JAX/XLA.
"""

from __future__ import annotations

import heapq
import json
import platform
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

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
    software_version: str = "0.1.0"
    environment: str = platform.platform()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @staticmethod
    def from_json(value: str) -> Certificate:
        return Certificate(**json.loads(value))


def verify_certificate(cert: Certificate, tolerance: float = 0.0) -> bool:
    """Replay bounds and prove that terminal boxes cover the declared domain."""
    expr = Expr.from_dict(cert.expression)
    if (
        not cert.active_regions
        or tolerance < 0
        or cert.absolute_gap < 0
        or not np.all(
            np.isfinite([cert.upper_bound, cert.lower_bound, cert.absolute_gap, tolerance])
        )
    ):
        return False
    domain_lo = np.asarray(cert.domain_lower, dtype=np.float64)
    domain_hi = np.asarray(cert.domain_upper, dtype=np.float64)
    incumbent = np.asarray(cert.incumbent_x, dtype=np.float64)
    if (
        domain_lo.ndim != 1
        or domain_lo.shape != domain_hi.shape
        or incumbent.shape != domain_lo.shape
        or not np.all(np.isfinite(domain_lo))
        or not np.all(np.isfinite(domain_hi))
        or np.any(domain_lo >= domain_hi)
        or np.any(incumbent < domain_lo)
        or np.any(incumbent > domain_hi)
    ):
        return False
    lbs = []
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for r in cert.active_regions:
        lo = np.asarray(r["lo"], dtype=np.float64)
        hi = np.asarray(r["hi"], dtype=np.float64)
        if (
            lo.shape != domain_lo.shape
            or hi.shape != domain_lo.shape
            or np.any(lo < domain_lo)
            or np.any(hi > domain_hi)
            or np.any(lo >= hi)
        ):
            return False
        iv = expr.interval(lo, hi)
        lbs.append(float(iv.lo))
        if not np.isfinite(r["lower_bound"]) or float(iv.lo) + tolerance < r["lower_bound"]:
            return False
        boxes.append((lo, hi))
    if not _boxes_cover_domain(boxes, domain_lo, domain_hi):
        return False
    point = expr.interval(incumbent, incumbent)
    return (
        float(point.hi) <= cert.upper_bound
        and min(lbs) >= cert.lower_bound - tolerance
        and cert.upper_bound - cert.lower_bound <= cert.absolute_gap + tolerance
    )


def _boxes_cover_domain(
    boxes: list[tuple[np.ndarray, np.ndarray]], domain_lo: np.ndarray, domain_hi: np.ndarray
) -> bool:
    """Verify exact volume equality and pairwise interior disjointness."""
    for i, (lo_i, hi_i) in enumerate(boxes):
        for lo_j, hi_j in boxes[i + 1 :]:
            if np.all(np.maximum(lo_i, lo_j) < np.minimum(hi_i, hi_j)):
                return False

    def volume(lo: np.ndarray, hi: np.ndarray) -> Fraction:
        result = Fraction(1)
        for lower, upper in zip(lo, hi, strict=True):
            result *= Fraction.from_float(float(upper)) - Fraction.from_float(float(lower))
        return result

    covered = sum((volume(lo, hi) for lo, hi in boxes), Fraction(0))
    return covered == volume(domain_lo, domain_hi)


def certify(
    expr: Expr, lo: Any, hi: Any, gap: float, max_nodes: int, problem_id: str
) -> Certificate:
    """Interval branch-and-bound with midpoint incumbents and widest splits."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    mid = (lo + hi) / 2
    best_x, upper = mid, float(expr.interval(mid, mid).hi)
    counter, processed, pruned = 0, 0, 0
    terminal: list[tuple[float, np.ndarray, np.ndarray]] = []
    lower = float(expr.interval(lo, hi).lo)
    heap: list[tuple[float, int, np.ndarray, np.ndarray]] = [(lower, counter, lo, hi)]
    while heap and upper - heap[0][0] > gap and processed < max_nodes:
        lb, _, a, b = heapq.heappop(heap)
        processed += 1
        if lb >= upper:
            pruned += 1
            terminal.append((lb, a, b))
            continue
        d = int(np.argmax(b - a))
        m = (a[d] + b[d]) / 2
        children = [(a.copy(), b.copy()), (a.copy(), b.copy())]
        children[0][1][d], children[1][0][d] = m, m
        for ca, cb in children:
            cm = (ca + cb) / 2
            value = float(expr.interval(cm, cm).hi)
            if value < upper:
                upper, best_x = value, cm
            cl = float(expr.interval(ca, cb).lo)
            if cl < upper:
                counter += 1
                heapq.heappush(heap, (cl, counter, ca, cb))
            else:
                pruned += 1
                terminal.append((cl, ca, cb))
    leaves = [{"lo": a.tolist(), "hi": b.tolist(), "lower_bound": lb} for lb, _, a, b in heap]
    leaves.extend({"lo": a.tolist(), "hi": b.tolist(), "lower_bound": lb} for lb, a, b in terminal)
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
