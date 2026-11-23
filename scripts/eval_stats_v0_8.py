"""Statistics and decision rules for the v0.8 main-track evaluation.

Pre-registration ``docs/research/v0.8-preregistration.md`` §7–§8, with amendments 1, 2 and 4
in ``docs/research/v0.8-preregistration-amendments.md``. Pure functions; no data access.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import beta, rankdata

GRID = np.arange(10, 801, 10)  # amendment 1: 80 budgets
MAX_UNITS = 800
F0_BUDGETS = (25, 50, 100, 200, 400, 800)
DRAWS = 10_000
_CHUNK = 250


# ---- per-instance summaries ------------------------------------------------------------


def unit_area(cost: float | None) -> float:
    """Amendment 1: share of the 80 grid budgets at which the instance has succeeded."""
    if cost is None or cost > MAX_UNITS:
        return 0.0
    return float(np.mean(GRID >= cost))


def time_area(t: float, horizon: float) -> float:
    """Amendment 1 wall axis: share of the grid {j·T/80 : j = 1..80} at or after ``t``."""
    if not np.isfinite(t):
        return 0.0
    grid = horizon * np.arange(1, 81) / 80.0
    return float(np.mean(grid >= t))


def success_within(units: list[int | None], budget: float) -> np.ndarray:
    return np.array([u is not None and u <= budget for u in units])


def sequential_cost(per_start: list[int | None], cap: int) -> int | None:
    """Amendment 1: start k begins at ``k·cap`` and may use ``min(cap, 800 − k·cap)`` units;
    the unit of the first success, or None."""
    for k, u in enumerate(per_start):
        begin = k * cap
        if begin >= MAX_UNITS:
            break
        if u is not None and u <= min(cap, MAX_UNITS - begin):
            return begin + u
    return None


def b95(units: list[int | None]) -> int | None:
    """Smallest F0-grid budget with at least 95% success (reported only)."""
    for b in F0_BUDGETS:
        if success_within(units, b).mean() >= 0.95:
            return b
    return None


# ---- intervals -------------------------------------------------------------------------


def crossed_bootstrap(matrix: Any, draws: int = DRAWS, seed: int = 0) -> dict[str, Any]:
    """Mean of ``matrix[seed, instance]`` with a 95% two-level crossed percentile bootstrap:
    each draw resamples the seeds and the instances independently (§7; the project's
    convention, as in v0.6 and v0.7). A single row is an instance-only bootstrap."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 1:
        m = m[None]
    if not np.all(np.isfinite(m)):
        raise ValueError("nonfinite paired values must be resolved before the bootstrap")
    rng = np.random.default_rng(seed)
    n_seeds, n = m.shape
    out = np.empty(draws)
    for start in range(0, draws, _CHUNK):
        k = min(_CHUNK, draws - start)
        s = rng.integers(0, n_seeds, (k, n_seeds))
        t = rng.integers(0, n, (k, n))
        row_means = m[:, t].mean(axis=2)  # (seeds, k)
        out[start:start + k] = np.take_along_axis(row_means.T, s, axis=1).mean(axis=1)
    return {"mean": float(m.mean()), "ci95": [float(np.quantile(out, 0.025)),
                                              float(np.quantile(out, 0.975))],
            "n_seeds": int(n_seeds), "n_instances": int(n), "draws": draws, "seed": seed}


def clopper_pearson(x: float, n: int, level: float = 0.95) -> list[float]:
    """Exact two-sided interval; ``x`` may be fractional (a sum of seed-mean rates)."""
    a = (1 - level) / 2
    lo = 0.0 if x <= 0 else float(beta.ppf(a, x, n - x + 1))
    hi = 1.0 if x >= n else float(beta.ppf(1 - a, x + 1, n - x))
    return [lo, hi]


def auroc(negative: Any, positive: Any) -> float:
    """P(positive score > negative score) + ½ P(tie) (Mann–Whitney)."""
    neg, pos = np.asarray(negative, float), np.asarray(positive, float)
    ranks = rankdata(np.concatenate([neg, pos]))
    r_pos = ranks[neg.size:].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def logistic_fit(x: Any, y: Any, iterations: int = 50) -> tuple[float, float]:
    """Maximum-likelihood ``P(y) = σ(a + b·x)`` by Newton's method (C9, reported only)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    design = np.stack([np.ones_like(x), x], axis=1)
    w = np.zeros(2)
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-(design @ w)))
        grad = design.T @ (y - p)
        hess = design.T @ (design * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


# ---- decisions (per family) ------------------------------------------------------------


def decide_superiority(ci: list[float]) -> str:
    """C1, C6, C7: supported if lower bound > 0; not supported if upper bound ≤ 0."""
    if ci[0] > 0:
        return "supported"
    if ci[1] <= 0:
        return "not supported"
    return "inconclusive"


def decide_c3(success_ci: list[float], cost_ci: list[float], margin: float = 0.02) -> str:
    """C3 (amendment 2 cost axis): better on one axis (lower bound > 0) and non-inferior
    on the other (lower bound ≥ −margin). Not supported if neither axis is better, or if
    either is inferior (interval entirely below −margin, amendment 4)."""
    def better(ci: list[float]) -> bool:
        return ci[0] > 0

    def non_inferior(ci: list[float]) -> bool:
        return ci[0] >= -margin

    def inferior(ci: list[float]) -> bool:
        return ci[1] < -margin

    if (better(success_ci) and non_inferior(cost_ci)) or \
            (better(cost_ci) and non_inferior(success_ci)):
        return "supported"
    if not (better(success_ci) or better(cost_ci)) or inferior(success_ci) or inferior(cost_ci):
        return "not supported"
    return "inconclusive"


def decide_c5(success_ci: list[float], silent_ci: list[float]) -> str:
    if success_ci[0] >= -0.01 and silent_ci[1] <= 0.01:
        return "supported"
    if success_ci[1] < -0.01 or silent_ci[0] > 0.01:
        return "not supported"
    return "inconclusive"


def decide_all_better(cis: list[list[float]]) -> str:
    """C2 (amendment 1): every bar on both axes has lower bound > 0; not supported if any
    upper bound < 0."""
    if all(ci[0] > 0 for ci in cis):
        return "supported"
    if any(ci[1] < 0 for ci in cis):
        return "not supported"
    return "inconclusive"


def decide_c4(cis: list[list[float]]) -> str:
    """C4 (amended): t_bar − t_A4 > 0 with lower bound > 0 for every bar; not supported if
    any bar's upper bound ≤ 0."""
    if all(ci[0] > 0 for ci in cis):
        return "supported"
    if any(ci[1] <= 0 for ci in cis):
        return "not supported"
    return "inconclusive"


def decide_c8(auroc_cis: list[list[float]]) -> str:
    if all(ci[0] > 0.5 for ci in auroc_cis):
        return "supported"
    if any(ci[1] <= 0.5 for ci in auroc_cis):
        return "not supported"
    return "inconclusive"


def decide_c9(ratio_ci: list[float]) -> str:
    if ratio_ci[0] >= 0.8:
        return "supported"
    if ratio_ci[1] < 0.8:
        return "not supported"
    return "inconclusive"


def decide_c10(ratio_ci: list[float]) -> str:
    if ratio_ci[1] < 0.05:
        return "supported"
    if ratio_ci[0] > 0.05:
        return "not supported"
    return "inconclusive"
