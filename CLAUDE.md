# NeuralOpt — Claude Code project instructions

## Mission
Build a compact, research-grade **learned optimizer** in JAX. From v0.6 onward, test whether **trajectory-aware curvature representation** helps the transformer learn transferable search directions, preconditioning and stochastic geometry. Do not build a solver dispatcher and do not introduce PINNs.

## Always inspect reality first
- Read `pyproject.toml`, `src/neuralopt/{features,model,optimizer,training,stochastic_geometry,hamiltonian,langevin,certification}.py`, `docs/research/transferable-geometry.md`, current tests and experiments. Confirm git branch/status/HEAD and PR state.
- Existing v0.5 negative result is authoritative until **new experiments** supersede it. Do not claim learned geometry beats validation-tuned fixed diagonal.
- Use existing JAX, testing, checkpointing, documentation and results conventions. Keep source changes minimal, not an architectural rewrite.

## Non-negotiable scientific rules
- NeuralOpt's transformer is the optimizer. Classical solvers may be used only as **benchmark or offline research references**, never as hidden inference backends.
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

## Research workflow
Read `docs/plans/neuralopt-v06-trajectory-curvature.md` and `docs/research/trajectory-curvature-experiments.md`. Use the local skills (see `SKILLS.md`). When installed, invoke Superpowers' brainstorming, writing-plans, test-driven-development and verification procedures instead of copying its source into this repository.
