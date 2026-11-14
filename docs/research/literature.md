# Literature audit

This audit informed the 0.1.0 design. Publication labels are conservative: the listed venue papers are peer reviewed; arXiv-only records are preprints unless a venue is named. Sources dated 2026 were supplied in the project brief and require bibliographic revalidation before scholarly publication.

## Learned optimization

Andrychowicz et al. (NeurIPS 2016) demonstrated coordinatewise recurrent optimizers trained by unrolling. Wichrowska et al. (ICML 2017) added hierarchical structure and broader tasks. Chen et al.'s JMLR 2022 survey formalized evaluation concerns and the gap between task-distribution success and general-purpose claims. VeLO (Metz et al., 2022 preprint) showed the value—and very large compute cost—of scale. Gärtner et al. (CVPR 2023) directly motivates attention over optimization trajectories. The NeurIPS 2022 universal hyperparameter-optimizer work uses transformers for transferable sequential decisions. B2Opt (AAAI 2025) addresses small-budget black-box search; MetaBox-v2 (2025 preprint) supplies an evolutionary-optimization benchmark ecosystem. The supplied 2026 gradient-history and Celo2 preprints are relevant recent claims, but their October 2026/arXiv and February 2026 status should be checked against canonical records before citation.

Common findings are sensitivity to training distributions, horizon truncation, feature normalization, meta-training cost, and incomplete comparisons after accounting for objective calls and wall time. Learned optimizers can exploit distributions; No Free Lunch results rule out uniform dominance over all objectives.

## Representations and theory

Deep Sets establishes permutation-invariant set aggregation; Set Transformers add attention-based interactions. Von Oswald et al. (ICML 2023) construct transformer mechanisms that implement gradient-descent-like in-context updates, but this does not show that an arbitrary trained transformer discovers a robust numerical optimizer. Variable-cardinality attention, latent task tokens, coordinate permutations, normalized dimensionless features, and low-rank curvature representations are therefore useful inductive biases rather than guarantees. Bilevel unrolling provides gradients through training trajectories but creates truncation bias and memory cost. Generalization across dimension, horizon, rotations, and conditioning must be measured separately.

## Certification

Classical interval branch-and-bound derives admissible regional lower bounds, feasible upper bounds, and prunes only by proof. Natural interval extensions suffer dependency overestimation; centered forms, interval gradients/Hessians, interval Newton/Krawczyk operators, McCormick relaxations, and Taylor models can tighten bounds. IEEE 1788 specifies interval semantics and decorations; ordinary JAX/XLA arithmetic does not provide directed rounding. `jax_verify` propagates certified relaxations mainly for neural-network verification. `jaxglobal` explores JAX-oriented global optimization. Neither makes arbitrary JAX programs automatically certifiable. The supplied PNAS Nexus 2026 GPU interval-global-optimization DOI should be authenticated before its results are quoted.

The implemented backend consequently uses a tiny serializable expression subset and host binary64 `nextafter` expansion. It fails closed for arbitrary JAX programs. This is more limited than JAXPR interval interpretation, interval Newton, or a conforming IEEE 1788 library, but its claim is inspectable.

## Evaluation

COCO/BBOB provides target-based, multi-instance benchmarking and discourages conclusions from a few analytic functions. bbob-jax helps batch/JIT evaluation, while MetaBox-v2 focuses learned evolutionary optimizers. Google `learned_optimization` provides tasks and learned-optimizer research infrastructure rather than a drop-in universally superior baseline. Sound studies need held-out families and instances, matched evaluation budgets, compile-time separation, paired seeds, confidence intervals, failures, and training cost.

## References

- Andrychowicz et al. 2016, arXiv:1606.04474.
- Wichrowska et al. 2017, *ICML*.
- Chen et al. 2022, *JMLR* 23(189).
- Metz et al. 2022, arXiv:2211.09760.
- Gärtner et al. 2023, *CVPR*.
- Chen et al. 2022, *NeurIPS*.
- von Oswald et al. 2023, *ICML*.
- Zaheer et al. 2017, *NeurIPS* (Deep Sets).
- Lee et al. 2019, *ICML* (Set Transformer).
- Moore, Kearfott, and Cloud 2009, *Introduction to Interval Analysis*.
- IEEE Std 1788-2015.
- Wolpert and Macready 1997, *IEEE Transactions on Evolutionary Computation*.

