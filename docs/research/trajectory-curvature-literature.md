# Trajectory curvature — primary literature and translation

**Use these as mathematical/architectural inputs, not copy-and-paste algorithms.** Checked 21 Nov 2026. The supplied arXiv link is not a proposal to build PINNs.

| Source | Transferable contribution | NeuralOpt application / novelty caveat |
| --- | --- | --- |
| Kiyani et al., *Optimizing the Optimizer for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks*, arXiv:2501.16371v6 (2026 revision), https://arxiv.org/html/2501.16371v6 | Historical secants; self-scaled BFGS/Broyden; step/line-search choices; precise benchmarking | Study scale normalization and curvature-pair statistics on **arbitrary JAX objectives**. Ignore PINN/PIKAN/DeepONet architectures and PDE losses. Not evidence that a transformer or learned matrix wins by itself. |
| Gärtner et al., *Transformer-Based Learned Optimization* (Optimus), CVPR 2023, https://openaccess.thecvf.com/content/CVPR2023/html/Gartner_Transformer-Based_Learned_Optimization_CVPR_2023_paper.html | Learned per-coordinate transformer features, rank-one preconditioner updates, variable dimensionality | Strong novelty baseline: transformer curvature and rank updates already exist. Compare matched compact learned history rather than repeating Optimus wholesale. |
| Lifshitz et al., *L-SR1: Learned Symmetric-Rank-One Preconditioning*, ICML 2026, https://proceedings.mlr.press/v306/lifshitz26a.html | Projection-guided secant regularization and learned PSD preconditioners | Audit PGSM and how it preserves definiteness; derive a compatible soft secant-consistency ablation, **do not claim a new principle**. |
| *Learning to Optimize Quasi-Newton Methods* (LODO), https://arxiv.org/abs/2210.06171 | Online learning of an implicit inverse-Hessian action without offline meta-training | Contrast pretrained transfer versus within-task online adaptation and cost. Treat LODO as a competitor, not a dependency. |
| Lee et al., *Set Transformer*, ICML 2019, https://proceedings.mlr.press/v97/lee19d.html | Set attention with permutation invariance/equivariance | Use masked attention across secant records; **temporal order is not exchangeable**. Coordinate permutations should permute coordinate outputs. |
| Vicol et al., *Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies*, ICML 2021, https://proceedings.mlr.press/v139/vicol21a.html | Meta-gradient bias/variance tradeoff in long unrolls | Consider only if truncated meta-gradients prove a bottleneck; no heavyweight redesign by default. |
| *Approximating Hessian matrices using Bayesian inference: a new approach for quasi-Newton methods in stochastic optimization* (2024), https://doi.org/10.1080/10556788.2024.2339226 | Noisy secant residuals and bounded-eigenvalue Hessian inference | Model uncertainty/reliability in noisy `(s,y)` pairs; do not mistake random gradient differences for exact Hessian actions. |
| *A Stochastic Quasi-Newton Method for Large-Scale Optimization* (2015), https://doi.org/10.1137/140954362 | Noise-robust stochastic curvature updates | Negative control for naive secant aggregation from independent noisy gradients. |
| *Learned Optimizers that Scale and Generalize* (2017), https://research.google/pubs/learned-optimizers-that-scale-and-generalize/ | Cross-scale generalization and hierarchical conditioning | Design variable-dimensional transfer protocol, not a claim of automatic mesh/rotation invariance. |

## Core algebra and caution

Let `u=2(x-lo)/(hi-lo)-1`, `x=lo + J(u+1)` with `J=diag((hi-lo)/2)`; `g_u=J^T g_x`. On quadratic `f(x)=0.5(x-x*)^T H_x (x-x*)`, `H_u=J^T H_x J`. From evaluated, coherent states, `s=Δu`, `y=Δg_u=H_u s`. A small history identifies `H_u` only on span of the observed displacements; full Hessian recovery is not implied. For inverse preconditioning, the corresponding ideal secant is `P y ≈ s`, **not** `P s≈y`. With indefinite objectives, `s^T y` can be negative; an SPD mass/mobility is still required for existing stochastic kernels.

## Research directions ordered by cost

1. **Masked historical secant statistics:** cheapest; test whether added information solves the v0.5 failure without attention.
2. **History reliability / robust secant loss:** detect degeneracy/noise, optionally use confidence-weighted consistency.
3. **Low-rank curvature sketch and conditional action:** explainable, matrix-free; diagnose rank and flop budget.
4. **Tiny cross-attention over historical pairs:** conditional temporal representation with cross-coordinate features; prove by ablation whether attention adds value.
5. **Optional JAX HVP probes:** accurate local curvature, but each probe is charged; do not use them as free privileged inference information.
6. **Stochastic geometry transfer:** reuse existing HMC/Langevin/BAOAB if earlier gates show improvement; preserve mass, mobility, noise conventions and source reproducibility.

Novelty is **not** any individual bullet. A publishable claim would require a new validated combination of observation-limited trajectory encoding, cost-efficient transfer, and demonstrable held-out gains over fixed diagonal and strong learned/second-order baselines.
