# NeuralOpt — Claude Code project instructions

If HANDOFF.md exists, read it first.

## Mission
NeuralOpt studies **where learning helps numerical optimization**, in JAX.
- **From v0.8, the learned component is a start predictor** trained on a problem family `f(θ; c)`. Refinement uses **NeuralOpt's own pure-JAX L-BFGS**, counted on the same objective/gradient budget.
- **External solvers (SciPy, Optimistix) remain benchmark references only**, never inference backends.
- **The learned-step transformer remains available as a research mode.** The v0.5–v0.7 negative results stay authoritative for it.
- All other rules below (no uncounted evaluations, frozen artifacts, pre-registration, honest negative results, independent certification) are unchanged. Do not build a solver dispatcher and do not introduce PINNs.

## Always inspect reality first
- Read `pyproject.toml`, `src/neuralopt/{features,model,optimizer,training,compiled,stochastic_geometry,hamiltonian,langevin,certification}.py`, `docs/research/{transferable-geometry,trajectory-curvature-v0.6,gram-space-v0.7}.md`, current tests and experiments. Confirm git branch/status/HEAD and PR state.
- The v0.5–v0.7 negative results are authoritative until **new pre-registered experiments** supersede them. Do not claim learned geometry beats a validation-tuned fixed diagonal or the BB step.
- Use existing JAX, testing, checkpointing, documentation and results conventions. Keep source changes minimal, not an architectural rewrite.

## Non-negotiable scientific rules
- At inference, every step comes from NeuralOpt's own code: the learned start predictor, the owned pure-JAX L-BFGS, or the learned-step research mode. SciPy, Optimistix and other external solvers may be used only as **benchmark or offline research references** (for example, reference optima and training targets), never as hidden inference backends.
- The same candidate must supply each `(position, objective, gradient)` observation; never form secants from stale or un-evaluated states. No uncounted evaluations.
- Normalize positions and gradients in **the same coordinate system**; scale curvature consistently with objective/parameter units. Mask missing, degenerate, noisy and rejected pairs rather than faking zeros.
- Mass/mobility matrices remain SPD and use the existing diagonal + low-rank structure unless evidence justifies a compatible alternative; signed negative curvature belongs in features/diagnostics, not an invalid SPD matrix.
- No PINNs, PIKANs, PDE collocation losses, DeepONet training, or solver integration for their sake. The supplied Kiyani et al. paper contributes secant/self-scaling theory and benchmarking lessons only.
- HMC, Langevin, Fokker–Planck diagnostics, function-space work and independent certification must continue working. Neural/stochastic predictions never certify global optimality.
- Never inflate training with leaked test objectives or tune fixed baselines on final held-out cases. Match objective/gradient/HVP counts and wall-clock, including compilation.
- Prevent silent checkpoint incompatibility: v0.1–v0.5 assets must load identically. Introduce versioned opt-in trajectory feature/model schema if input shape changes.
- Do not claim a rigorous proof, statistically significant advantage or novel method without corresponding verified evidence.

## Working discipline
1. Discover existing implementation and reproduce baseline.
2. State falsifiable hypothesis, minimal design, test matrix and resource budget.
3. Use test-first incremental implementation; one gate at a time.
4. Preserve frozen checkpoints/results; new experiments get versioned artifacts.
5. Run fast tests during iteration; full `uv run pytest -q`, `uv run ruff check .`, `uv run pyright`, `uv run zensical build`, and packaging checks before marking a gate done (adapt to installed project environments).
6. Report measured changes, negative results, uncertainty and unfinished gates.
7. Never merge, publish, force-push or delete unrelated work without explicit permission. A PR may be prepared after validation when authorized.
8. **v0.8 release freeze:** no v0.8.0 tag, PyPI upload or GitHub release until every claim in `docs/research/v0.8-claims.md` has a pre-registered result and the user has reviewed the register and approved the release. README, docs, CHANGELOG and release notes describe a v0.8 claim as "under test" until its status is "supported".
9. **Long runs:** any run expected to exceed 10 minutes runs in a named tmux (or screen) session, never as a child of the agent's shell, with a git-ignored run directory `runs/<name>-<UTC stamp>/` holding `command.txt` (command, git SHA, uv.lock hash, UTC start), `log.txt`, an atomically rewritten `progress.json`, checkpoints for resume, `exit_code`, and `DONE` only on success. Update `HANDOFF.md` (repo root, git-ignored) before and after the run. Versioned results still go to `results/` when the report is written.

## Research workflow
Read `docs/plans/neuralopt-v08-amortized-families.md` and the claims register `docs/research/v0.8-claims.md` (earlier: `docs/plans/neuralopt-v06-trajectory-curvature.md`). Use the local skills (see `SKILLS.md`). When installed, invoke Superpowers' brainstorming, writing-plans, test-driven-development and verification procedures instead of copying its source into this repository.
