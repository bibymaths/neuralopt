"""EXPLORATORY, NOT PRE-REGISTERED, feeds no decision: what did training contribute?

Evaluates gram-scalar and gram-full at their initial (untrained) weights, i.e. the
``init_gram_model(PRNGKey(seed), config)`` parameters that ``train_batched`` starts from,
on the same frozen test tasks at budget 100, and reports trained − untrained paired
intervals (paired by seed) on the curvature domain and the guard families. Trained rows
are read from ``results/gram-space-stage2-v0.7.0.json``. Requested 2026-11-22 because
gram-scalar's training loss was flat from the start.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ["JAX_ENABLE_X64"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402

from neuralopt.gram import init_gram_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gram_space_v0_7 as protocol  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("gram-scalar", "gram-full")


def main() -> None:
    assert jax.config.jax_enable_x64
    prereg = protocol.check_preregistration()
    stage2 = json.loads((ROOT / "results" / "gram-space-stage2-v0.7.0.json").read_text())
    trained = [r for r in stage2["rows"] if r["budget"] == protocol.PRIMARY_BUDGET
               and r["arm"] in (*ARMS, "bb")]
    untrained_params = {arm: [(seed, init_gram_model(jax.random.PRNGKey(seed),
                                                     protocol.ARMS[arm]))
                              for seed in protocol.SEEDS] for arm in ARMS}
    rows = protocol.evaluate_rows(protocol.test_specs(), untrained_params,
                                  budgets=(protocol.PRIMARY_BUDGET,), include_tf_h8=False,
                                  include_lbfgsb=False)
    untrained = [{**r, "arm": f"{r['arm']}-untrained"} for r in rows if r["arm"] in ARMS]
    combined = trained + untrained
    groups = {"curvature": protocol.CURVATURE, **{fam: (fam,) for fam in protocol.GUARD}}
    out = {
        "label": "EXPLORATORY - not pre-registered - feeds no decision",
        "preregistration_sha256": prereg,
        "test_split_sha256": protocol.split_hash(protocol.test_specs()),
        "initial_weights": "init_gram_model(jax.random.PRNGKey(seed), config), seeds 0-4",
        "trained_minus_untrained": {
            arm: {name: protocol.paired_interval(protocol.difference(
                combined, arm, f"{arm}-untrained", labels)) for name, labels in groups.items()}
            for arm in ARMS},
        "untrained_minus_bb": {
            arm: {name: protocol.paired_interval(protocol.difference(
                combined, f"{arm}-untrained", "bb", labels)) for name, labels in groups.items()}
            for arm in ARMS},
        "per_family_untrained": protocol.per_family(untrained + [
            r for r in trained if r["arm"] == "bb"], [f"{a}-untrained" for a in ARMS])[
            "mean_log10_regret_floored"],
        "rows": untrained,
    }
    path = ROOT / "results" / f"gram-untrained-diagnostic-v{protocol.VERSION}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
