"""Check that the evaluation harness reproduces the frozen per-instance validation data
(F0 instances, C2 validation starts, the validation choices) on validation instances.

Usage: python scripts/check_eval_reproduction_v0_8.py RUN_DIR
Comparisons are made on instances whose f* agrees with the frozen F0 f* to 1e-12 relative
(the harness's f* also includes the full-budget SciPy and learned arms, so it can only be
lower; differences at rounding level cannot change a success decision).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    run = Path(sys.argv[1])
    inst = {r["family"]: r for r in json.loads((ROOT / "results" / "f0-instances-v0.8.0.json")
                                               .read_text())["families"]}
    c2 = {r["family"]: r for r in json.loads((ROOT / "results" / "c2-validation-v0.8.0.json")
                                             .read_text())["families"]}
    ch = {r["family"]: r for r in json.loads((ROOT / "results" / "v0.8-validation-choices.json")
                                             .read_text())["families"]}
    report, ok = {}, True
    for fam_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        name = fam_dir.name
        rows = [r for p in sorted((fam_dir / "units").glob("*.json"))
                for r in json.loads(p.read_text())["row"]["rows"]]
        pi = inst[name]["per_instance"]
        same = [r for r in rows if abs(r["f_star"] - pi["f_star"][r["index"]])
                <= 1e-12 * abs(pi["f_star"][r["index"]])]
        checks = {}
        for arm, key in (("A0", "A0"), ("A3", "A3"), ("A1", "A1"), ("A1_full", "A1_full")):
            checks[arm] = all(r[arm]["u"] == pi[key]["units_to_success"][r["index"]] for r in same)
        for arm in ("A2o", "A2_full"):
            per = c2[name]["per_start_units_to_success"][arm]
            checks[arm] = all(r[arm] == [per[k][r["index"]] for k in range(16)] for r in same)
        runs = ch[name]["runs"]
        loss = ch[name]["chosen_loss"]
        # the harness computes starts one instance at a time, validate in a batch of 200;
        # the forward pass differs at rounding level between batch sizes, so A4 is compared
        # by its success at B (required equal); units and refined fit are reported
        pairs = [(r[f"learned_{loss}"], runs[f"{loss}/seed={s}"], r["index"], s)
                 for r in same for s in range(5)]
        checks["A4_fit_at_B_max_relative_difference"] = max(
            abs(a["fit_at_B"][s] - v["fit_at_B"][i]) / abs(v["fit_at_B"][i])
            for a, v, i, s in pairs)
        checks["A4_success_at_B"] = all(
            (a["u"][s] is not None and a["u"][s] <= ch[name]["matched_budget"])
            == v["success_at_B_per_instance"][i] for a, v, i, s in pairs)
        checks["A4_units_exact_share"] = sum(a["u"][s] == v["units_to_success"][i]
                                             for a, v, i, s in pairs) / len(pairs)
        checks["A4_conf_top"] = all(abs(r[f"learned_{loss}"]["conf_top"][s] -
                                        runs[f"{loss}/seed={s}"]["confidence_top"][r["index"]])
                                    < 1e-9 for r in same for s in range(5))
        checks["reference_reproduced_share"] = sum(r["reference_reproduced"] for r in rows) \
            / len(rows)
        cpu = sum(json.loads(p.read_text())["row"]["cpu_seconds"]
                  for p in (fam_dir / "units").glob("*.json"))
        report[name] = {"instances": len(rows), "compared (same f*)": len(same),
                        "checks": checks, "cpu_seconds_per_instance": cpu / len(rows)}
        ok &= all(v for k, v in checks.items()
                  if k not in ("reference_reproduced_share", "A4_units_exact_share",
                                "A4_fit_at_B_max_relative_difference"))
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
