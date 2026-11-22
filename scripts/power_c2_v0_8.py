"""Power check for the amended C2 (pre-registration amendment 1): continuous cost to
success, summarized by the per-instance area under the success-versus-budget step function
on the grid {10, 20, ..., 800} units. Harness only; validation data only.

Inputs: ``results/f0-instances-v0.8.0.json`` (A0, A3 units to success, f*) and
``results/c2-validation-v0.8.0.json`` (units to success of each of the 16 Sobol starts for
owned L-BFGS and full-budget SciPy).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

GRID = np.arange(10, 801, 10)  # 80 budgets
MAX_UNITS = 800
CAPS = (25, 50, 100, 200, 400, 800)  # per-start caps tried on validation (multistart)
EFFECT = 0.05  # 5% of the grid: about 40 units of restricted mean cost
K_POWER = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)


def area(cost: float | None) -> float:
    """Share of grid budgets at which the instance has already succeeded (0 if never)."""
    if cost is None or cost > MAX_UNITS:
        return 0.0
    return float(np.mean(GRID >= cost))


def sequential_cost(per_start: list[int | None], cap: int) -> int | None:
    """Starts run one after another; start k begins at ``k · cap`` and may use
    ``min(cap, 800 − k · cap)`` units. Returns the unit of the first success, or None."""
    for k, u in enumerate(per_start):
        begin = k * cap
        if begin >= MAX_UNITS:
            break
        if u is not None and u <= min(cap, MAX_UNITS - begin):
            return begin + u
    return None


def family(row: dict[str, Any], starts: dict[str, Any]) -> dict[str, Any]:
    pi = row["per_instance"]
    n = len(pi["f_star"])
    a = {"A0": np.array([area(c) for c in pi["A0"]["units_to_success"]]),
         "A3": np.array([area(c) for c in pi["A3"]["units_to_success"]])}
    caps = {}
    for arm in ("A2o", "A2_full"):
        per = starts["per_start_units_to_success"][arm]  # [start][instance]
        best = None
        for cap in CAPS:
            areas = np.array([area(sequential_cost([per[k][i] for k in range(len(per))], cap))
                              for i in range(n)])
            if best is None or areas.mean() > best[1].mean():
                best = (cap, areas)
        assert best is not None
        caps[arm], a[arm] = best
    better = max(("A2o", "A2_full"), key=lambda k: a[k].mean())
    proxies = {"vs A0 (proxy A3 − A0)": a["A3"] - a["A0"],
               f"vs better multistart {better} (proxy A3 − {better})": a["A3"] - a[better],
               "vs A3 (proxy A0 − A3)": a["A0"] - a["A3"]}
    out = {}
    for label, d in proxies.items():
        sd = float(np.std(d, ddof=1))
        out[label] = {"mean_difference": float(d.mean()), "sd": sd,
                      "n_required": max(1, math.ceil((K_POWER * sd / EFFECT) ** 2))}
    return {"family": row["family"], "mean_area": {k: float(v.mean()) for k, v in a.items()},
            "chosen_per_start_cap": caps, "better_multistart": better, "comparisons": out,
            "n_required_max": max(v["n_required"] for v in out.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instances", type=Path, default=Path("results/f0-instances-v0.8.0.json"))
    parser.add_argument("--starts", type=Path, default=Path("results/c2-validation-v0.8.0.json"))
    parser.add_argument("--output", type=Path, default=Path("results/power-c2-v0.8.0.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; results are never overwritten")
    rows = {r["family"]: r for r in json.loads(args.instances.read_text())["families"]}
    starts = {r["family"]: r for r in json.loads(args.starts.read_text())["families"]}
    report = {"schema": "neuralopt-power-c2/v0.8", "grid": [int(GRID[0]), int(GRID[-1]), 10],
              "effect": EFFECT, "caps": CAPS,
              "families": [family(rows[name], starts[name]) for name in rows]}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    for f in report["families"]:
        print(f["family"], f["mean_area"], f["chosen_per_start_cap"], f["n_required_max"])
        for label, v in f["comparisons"].items():
            print("   ", label, {k: round(x, 3) for k, x in v.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
