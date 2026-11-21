---
name: neuralopt-audit
description: Audit the current NeuralOpt repository, v0.5 checkpoint compatibility, research artifacts, and pending trajectory-curvature work before implementing changes.
---

1. `git status --short --branch; git log -5 --oneline`; inspect `pyproject.toml` and actual code/tests (especially `features.py`, `model.py`, `optimizer.py`, `training.py`, `stochastic_geometry.py`, `certification.py`).
2. Read `docs/research/transferable-geometry.md`, `docs/research/trajectory-diagnostics.md`, and the matching v0.5 results. Verify the fixed-diagonal baseline and negative learned-geometry result; detect any newer commits and update understanding accordingly.
3. Inspect checkpoint deserialization and parameter-shape contracts. Specifically identify the 5-feature embedding and every API that assumes it.
4. Map state/evaluation information: which accepted/rejected states have a correctly evaluated objective and gradient? Where is candidate history recorded? How many gradients are counted?
5. Produce a short **as-is vs target** table, interface inventory, tests to preserve, risk register, and no-modification audit report. Avoid claiming outputs or tests were run unless you actually ran them.
6. Only then call `/curvature-theory` and propose a minimal test-first plan.
