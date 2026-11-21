---
paths:
  - "scripts/**"
  - "results/**"
  - "docs/research/**"
  - "tests/**"
---
# Benchmark and reporting integrity

- Validation-tune classical/fixed-diagonal controls on validation only; freeze all configurations before final held-out evaluation.
- Report matched **objective, gradient, HVP, stochastic-step and candidate** counts (not a vague shared budget); include model/training expense and cold/warm runtime separately.
- Compare history lengths 0/1/4/8, history-only feature baseline, matched MLP, current single-token transformer, L-BFGS/self-scaled Broyden references, and new trajectory model where applicable.
- Use paired random seeds and independent task instances; hold out whole families/rotations/dimensions (the v0.5 unseen 7/8/10 benchmark is historical evidence, not a fresh discovery set).
- Do not tune against final test data or overwrite `results/*v0.5.0*` or frozen checkpoint artifacts.
- Use positive paired log-regret differences as worse than reference; present paired confidence intervals, per-family results, failure rates, time-to-target and missing results honestly.
- Negative findings and cost regressions are first-class experimental results.
