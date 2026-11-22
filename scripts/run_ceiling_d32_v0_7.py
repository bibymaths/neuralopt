"""Oracle capacity ceiling at d = 32 on validation seeds (no training, no test tasks).

Frozen loop (halving, K = 10, float64, budget 100) with history-v2 (float64) masks;
rotated kappa = 100 and 1000, 4 tasks each, validation seeds 8000+. SciPy L-BFGS-B
(maxcor 10) is reported for context with paired intervals over tasks.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ["JAX_ENABLE_X64"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

from neuralopt.task_families import make_task, quadratic_objective  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_loop_rule_ablation_v0_7 import lbfgsb  # noqa: E402
from run_mask_diagnostic_v0_7 import FLOOR, make_loop, paired  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    assert jax.config.jax_enable_x64
    loop = make_loop("float64")
    out = {}
    for condition in (100.0, 1000.0):
        oracle, scipy = [], []
        for r in range(4):
            task = make_task("rotated", condition, 32, 8000 + 100 * int(np.log10(condition)) + r)
            f0 = float(quadratic_objective(task.x0, task.theta))
            o = loop(task.theta, task.x0, task.lo, task.hi)
            oracle.append(float(np.log10(max(float(o["value"]), 0.0) / f0 + FLOOR)))
            best, _ = lbfgsb(task, 10)
            scipy.append(float(np.log10(max(best, 0.0) / f0 + FLOOR)))
        oracle_a, scipy_a = np.array(oracle), np.array(scipy)
        out[f"kappa_{int(condition)}"] = {
            "oracle_log10_regret": oracle, "lbfgsb_m10_log10_regret": scipy,
            "oracle_median": float(np.median(oracle_a)),
            "lbfgsb_median": float(np.median(scipy_a)),
            "oracle_reached_1e-6": f"{int(np.sum(oracle_a <= -6))}/4",
            "paired_oracle_minus_lbfgsb": paired(oracle_a - scipy_a),
        }
    (ROOT / "results" / "oracle-ceiling-d32-v0.7.0.json").write_text(json.dumps(
        {"schema": "neuralopt-oracle-ceiling/v1", "dimension": 32, "seeds": "validation 8000+",
         "loop": "halving, K=10, float64 masks (history-v2), budget 100", "results": out},
        indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
