# NeuralOpt Claude Code starter kit — trajectory-aware curvature (21 Nov 2026)

**Purpose:** Continue the existing `bibymaths/neuralopt` project from the merged v0.5 source. This archive is an **overlay**, not a replacement repository. It contains no training weights or source-code changes.

## Quick start (Claude Code CLI)

1. Work in your current local checkout of `bibymaths/neuralopt`; make sure it is clean and up-to-date. Verify the branch and merged v0.5 state; do not assume `/workspace/neuralopt` exists on your machine.
2. Extract the archive into the repository root **without overwriting existing files** (e.g., `unzip -n neuralopt_claude_starter_kit.zip -d /path/to/neuralopt`). Inspect every destination before copying if you already use Claude customization.
3. In the repository root run `claude` in the terminal. Claude Code automatically reads `CLAUDE.md` and discovers the project skills under `.claude/skills/`.
4. Optional **one-time explicit install in Claude Code**: `/plugin install superpowers@claude-plugins-official`. This brings the official Superpowers development-methodology plugin; it is not bundled or copied into this kit. Restart Claude Code if requested. Use its brainstorming → writing-plans → executing-plans → verification workflow. Do not install any other agent platform.
5. Paste `MASTER_PROMPT.md` into Claude Code or call `/neuralopt-audit` followed by `/curvature-theory` and `/trajectory-curvature`.
6. Require a concrete design and experiment plan before code changes. No release/merge/push by default.

## Roles of the files

- `CLAUDE.md`: compact always-on mission and repository invariants.
- `SKILLS.md`: human-readable skill map; **not a special automatic loader**.
- `.claude/rules/`: mathematical correctness and benchmark constraints.
- `.claude/skills/*/SKILL.md`: actionable procedures and task prompts recognized by Claude Code.
- `.claude/agents/`: read-only numerical and evaluation reviewers.
- `docs/plans/`: staged v0.6+ implementation roadmap, not a claim that milestones exist.
- `docs/research/`: literature-to-implementation translation and prespecified tests.
- `MASTER_PROMPT.md`: single end-to-end request.

## Current repository facts (inspected 21 Nov 2026)

- Default branch `main`, `pyproject.toml` version `0.5.0`; PR #5 merged.
- `src/neuralopt/features.py` currently exposes `parameter_features` with **5 features per coordinate**, using normalized coordinates and a scale-aware feature.
- `src/neuralopt/model.py` initializes its embedding with `w(5, config.width)`; changing this blindly will break existing `.npz` checkpoints.
- `src/neuralopt/optimizer.py` supplies direct, Hamiltonian, overdamped Langevin, and underdamped variants. `certified` mode is separate and mathematically restricted.
- v0.5 report (`docs/research/transferable-geometry.md`) finds full/FP learned overdamped geometry significantly worse than validation-tuned fixed-diagonal on unseen dimensions 7/8/10.
- Existing native interval certification, stochastic geometry, Fokker–Planck diagnostics, and function-space utilities are assets. **Do not rewrite or retire them.**

## Important limits

The literature points to plausible research directions; none guarantees a universal improvement. The PDF supplied by the user is a PINN/PIKAN *testbed* for self-scaled quasi-Newton ideas. No PINN architecture, collocation loss, or PDE residual training enters NeuralOpt solely because of that paper.

Reviewed sources: arXiv:2501.16371v6 (Kiyani et al.); L-SR1 (ICML 2026); Optimus (CVPR 2023); LODO; Set Transformer; noisy secant/Hessian literature. See `docs/research/trajectory-curvature-literature.md`.
