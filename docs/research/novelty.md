# Novelty and falsification

The individual ingredients—transformers, learned updates, diagonal-plus-low-rank matrices, JAX, and interval branch-and-bound—are established. The candidate contribution is a compact experimental interface in which the same policy emits numerical updates and admissible search-order suggestions while certificate validity remains fully independent. This integration is a hypothesis, not established novelty or improved performance.

| System family | Learned direct update | Variable dimension | Native rigorous global bound | Neural guidance can affect proof validity | Evidence available here |
|---|---:|---:|---:|---:|---|
| L2L / scalable recurrent optimizers | Yes | Varies | No | N/A | Published studies |
| Transformer learned optimizers | Yes | Varies | No | N/A | Published studies |
| B2Opt / MetaBox learned black-box methods | Yes | Often | No | N/A | External studies |
| SciPy/JAXopt/Optimistix | No | Yes | No | N/A | Mature software |
| Interval global solvers / jaxglobal | No | Yes | Yes within subset | N/A | External software |
| neuralopt 0.1.0 | Yes | Yes | Yes under documented restricted assumptions | No | Local tests and demo only |

Reported gaps in prior work include distribution shift, meta-training cost, weak wall-time accounting, unavailable weights, limited dimension/horizon transfer, and no rigorous relation between policy confidence and correctness. This package does not yet resolve them.

H1 is falsified if a trained checkpoint fails to beat its random initialization on held-out families and dimensions under paired budgets. H2 is falsified if matched SciPy and learned baselines dominate objective quality and time across the prespecified task distribution. H3 is falsified if learned ordering does not reduce median processed nodes at the same certified gap; correctness must be identical either way. H4 is falsified if inference overhead or instability erases evaluation savings, or parameter count/compute is not competitive. Gate thresholds for a larger study are: at least 20 held-out instances per family, five seeds, median regret improvement of 10% with a bootstrap 95% interval excluding zero for the claimed distribution, no increase in failure rate, and certificate replay success in every certified run.

