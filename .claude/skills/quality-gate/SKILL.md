---
name: quality-gate
description: Verify NeuralOpt source minimalism, old checkpoints, numerical correctness, CI, documentation, packaging and research claim boundaries before a PR.
---

1. Compare changed files to the approved minimal plan; remove unneeded new modules or docs, not scientifically needed history/data.
2. Run targeted tests, then full `uv run pytest -q` (or repository-validated equivalent), Ruff, Pyright, Zensical, build and Twine checks. Report exactly what ran and what did not.
3. Run legacy checkpoint load + v0.5 deterministic regression + no-history equivalence + new history/JAX geometry tests + certification/replay tests.
4. Rerun relevant locked benchmark configs and verify no data leakage, correct counts, valid CIs, reproducible figure scripts.
5. Update relevant docs, `CHANGELOG.md` and artifact manifests. Do not assert improvements from training loss alone.
6. Produce changed-file manifest, tested SHAs, measured results with uncertainty, risk list, next gate. Never merge/release/push without permission.
