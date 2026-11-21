---
name: geometry-benchmarks
description: Preregister and execute matched-budget, out-of-distribution NeuralOpt curvature/geometry ablations and honest numerical performance reporting.
---

Read `docs/research/trajectory-curvature-experiments.md`. Freeze final-test tasks/seeds before training. Start with a representation-identification unit experiment on known Hessians; then held-out tasks with isotropic, rotated ill-conditioned, indefinite and multimodal objectives across dimensions (including unseen dimensions beyond training).

Comparators: v0.5 single observation, fixed-diagonal (validation tuned), deterministic secant/sketch reference, history-statistic MLP, small trajectory transformer; also independent SciPy quasi-Newton/self-scaled Broyden when available **only in the harness**. Do not introduce them into NeuralOpt inference.

Report paired log-regret difference (negative favors candidate), confidence intervals, per-family results, time-to-target, first-call and warm runtime, objective/gradient/HVP counts, history-memory and model latency, training cost, certification independence. Use multiple meta-training seeds, not merely 30 repeated test instances with one training seed.

Include negative outcomes and a stop decision if the simple baseline wins. Regenerate plots from versioned machine-readable artifacts; never overwrite v0.5. Submit results to `experiment-auditor` before claiming improvement.
