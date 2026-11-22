"""Power check for the v0.8 pre-registration (main track, C1-C10), from F0's per-instance
validation data (``results/f0-instances-v0.8.0.json``). Harness only; no learned arm exists
yet, so spreads come from proxies (pairs of real start strategies on validation).

Rule (pre-registration §9): every comparison is powered at 80% for its stated effect at the
two-sided 95% level, ``n = ((z_0.975 + z_0.8) · SD / δ)²``. For a paired binary difference
with true mean δ the spread is at least ``sqrt(δ (1 − δ))``, so proxy spreads are floored
there. Test sizes are raised to the largest requirement among a family's primary claims;
criteria are never loosened.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

K_POWER = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
MATCHED_B = {"sinusoid": 200, "cascade-clean": 400, "cascade-realistic": 800}
PRIMARY = {"sinusoid": ("C1", "C3-success", "C6"), "cascade-clean": ("C5-success",
                                                                     "C5-silent-miss"),
           "cascade-realistic": ("C2", "C7")}
EFFECT = {"superiority": 0.05, "C3-noninferiority": 0.02, "C5-noninferiority": 0.01}
GRID = (25, 50, 100, 200, 400, 800)


def n_required(sd: float, delta: float) -> int:
    return max(1, math.ceil((K_POWER * sd / delta) ** 2))


def floored_sd(diff: np.ndarray, delta: float) -> tuple[float, float]:
    empirical = float(np.std(diff, ddof=1))
    return empirical, max(empirical, math.sqrt(delta * (1 - delta)))


def success_at(units: list[Any], b: int) -> np.ndarray:
    return np.array([u is not None and u <= b for u in units], float)


def multistart_best(matrix: dict[str, Any], b: int) -> tuple[int, np.ndarray]:
    best_k, best = None, None
    for k, by_b in matrix.items():
        if str(b) in by_b:
            s = np.asarray(by_b[str(b)], float)
            if best is None or s.mean() > best.mean():
                best_k, best = int(k), s
    assert best is not None and best_k is not None
    return best_k, best


def silent_miss_n(rate: float, bound: float = 0.01, power: float = 0.8,
                  n_max: int = 20000) -> int | None:
    """Smallest n (multiple of 100) at which P(Clopper-Pearson upper <= bound) >= power,
    for a true silent-miss rate ``rate``."""
    from scipy.stats import beta, binom

    for n in range(100, n_max + 1, 100):
        # largest k with CP upper <= bound (CP upper via the beta quantile)
        k_ok = -1
        for k in range(0, n):
            upper = beta.ppf(0.975, k + 1, n - k)
            if upper <= bound:
                k_ok = k
            else:
                break
        if k_ok >= 0 and binom.cdf(k_ok, n, rate) >= power:
            return n
    return None


def b95_stability(units_or_success: Any, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Bootstrap the B95 of one arm (success-vs-budget curve, resampling instances)."""
    rng = np.random.default_rng(seed)
    curve = np.stack([units_or_success(b) for b in GRID])  # (budgets, instances)
    n = curve.shape[1]

    def b95(c: np.ndarray) -> int | None:
        ok = np.flatnonzero(c.mean(axis=1) >= 0.95)
        return int(GRID[ok[0]]) if ok.size else None

    point = b95(curve)
    draws = [b95(curve[:, rng.integers(0, n, n)]) for _ in range(n_boot)]
    same = float(np.mean([d == point for d in draws]))
    return {"b95": point, "bootstrap_share_equal_to_point": same,
            "bootstrap_values": sorted({d for d in draws if d is not None}) +
            (["none"] if any(d is None for d in draws) else [])}


def family_power(row: dict[str, Any]) -> dict[str, Any]:
    name, pi = row["family"], row["per_instance"]
    b = MATCHED_B[name]
    a0, a3 = success_at(pi["A0"]["units_to_success"], b), success_at(pi["A3"]["units_to_success"],
                                                                      b)
    k_o, a2o = multistart_best(pi["A2o"], b)
    k_f, a2f = multistart_best(pi["A2_full"], b)
    a2o_k4 = np.asarray(pi["A2o"]["4"][str(b)], float)
    sup = EFFECT["superiority"]
    out: dict[str, Any] = {"family": name, "matched_budget": b,
                           "rates_at_B": {"A0": a0.mean(), "A3": a3.mean(),
                                          f"A2o(K={k_o})": a2o.mean(),
                                          f"A2_full(K={k_f})": a2f.mean(),
                                          "A2o(K=4)": a2o_k4.mean()}}
    claims: dict[str, Any] = {}

    def add(claim: str, proxy: str, diff: np.ndarray, delta: float) -> None:
        emp, used = floored_sd(diff, delta)
        claims[claim] = {"proxy": proxy, "effect": delta, "sd_empirical": emp, "sd_used": used,
                         "n_required": n_required(used, delta)}

    add("C1", "A3 − A0 success at B", a3 - a0, sup)
    add("C3-success", "A3 − A0 success at B (non-inferiority margin 2 pp, true difference 0)",
        a3 - a0, EFFECT["C3-noninferiority"])
    add("C6", "A3 − A2o(K=4) success at B", a3 - a2o_k4, sup)
    add("C7", "A3 − A0 success at B (two A4 variants)", a3 - a0, sup)
    add("C5-success", "A0 − A2_full success at B (margin 1 pp, true difference 0)",
        a0 - a2f, EFFECT["C5-noninferiority"])
    for rate in (0.001, 0.0025, 0.005):
        claims[f"C5-silent-miss@{rate}"] = {
            "proxy": f"exact binomial; assumed true silent-miss rate {rate}",
            "rule": "Clopper-Pearson 95% upper bound <= 0.01 with probability >= 0.8",
            "n_required": silent_miss_n(rate)}
    claims["C5-silent-miss"] = claims["C5-silent-miss@0.0025"]
    p = max(float(a3.mean()), 0.05)
    n_q = n_required(math.sqrt((1 - p) / p), 0.2)
    claims["C9"] = {"proxy": f"A3 success at B (p = {p:.3f}); ratio 1 vs margin 0.8; "
                             "quartile size n/4 (conservative delta method)",
                    "n_required": 4 * n_q}
    n1 = n2 = 200  # OOD split vs the in-distribution test split (at least 200)
    se0 = math.sqrt((0.25 + (n1 - 1) / 12 + (n2 - 1) / 12) / (n1 * n2))  # Hanley-McNeil, A=0.5
    claims["C8"] = {"proxy": "Hanley-McNeil at AUROC 0.5, 200 OOD vs 200 test instances",
                    "auroc_detectable": round(0.5 + K_POWER * se0, 3)}
    stability = {
        "A0": b95_stability(lambda bb: success_at(pi["A0"]["units_to_success"], bb)),
        "A3": b95_stability(lambda bb: success_at(pi["A3"]["units_to_success"], bb)),
        "A2o(best K)": b95_stability(lambda bb: multistart_best(pi["A2o"], bb)[1]),
        "A2_full(best K)": b95_stability(lambda bb: multistart_best(pi["A2_full"], bb)[1]),
    }
    claims["C2/C3-cost"] = {"proxy": "bootstrap of each bar's B95 on validation (n = 200)",
                            "b95_stability": stability}
    claims["C4/C10"] = {"note": "timing claims: per-instance warm times are measured with "
                                "small relative noise; not powered by instance count"}
    # C2 (B95 ratios on the coarse grid) is checked by B95 stability, not by an n
    primary_n = [claims[c]["n_required"] for c in PRIMARY[name]
                 if c in claims and claims[c].get("n_required") is not None]
    test_size = max(200, int(math.ceil(max(primary_n) / 100.0) * 100)) if primary_n else 200
    out.update({"claims": claims, "primary_claims": list(PRIMARY[name]),
                "test_size": test_size,
                "c5_budget_note": "C5 is decided at the larger of the arms' B95; "
                                  "B is used as the proxy budget here"})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instances", type=Path,
                        default=Path("results/f0-instances-v0.8.0.json"))
    parser.add_argument("--output", type=Path, default=Path("results/power-v0.8.0.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; results are never overwritten")
    data = json.loads(args.instances.read_text())
    report = {"schema": "neuralopt-power/v0.8", "k_power": K_POWER, "effects": EFFECT,
              "matched_budgets": MATCHED_B,
              "families": [family_power(r) for r in data["families"]]}
    args.output.write_text(json.dumps(report, indent=1, default=float) + "\n")
    for fam in report["families"]:
        print(fam["family"], "test size", fam["test_size"],
              {c: v.get("n_required") for c, v in fam["claims"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
