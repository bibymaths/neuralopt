# Hypothesis experiment plan

All comparisons use paired task instances, seeds, dimensions, and evaluation budgets. H1 compares trained and untrained policies on held-out families and dimensions. H2 compares final and anytime regret, evaluations, failures, and wall time against an appropriate classical solver and available faithful learned baselines. H3 compares fixed widest-first branch ordering with model priority while replaying identical certificate rules. H4 records compilation, steady-state policy time, objective calls, parameter count, memory, and training cost. Ablations remove attention, history compression, gradients, low-rank factors, independent proposals, normalization augmentation, pretraining, and learned spatial guidance one factor at a time.

The bundled smoke experiment is insufficient for accepting any hypothesis: it verifies training and serialization on synthetic quadratics only.

