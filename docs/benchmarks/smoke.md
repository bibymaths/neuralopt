# Held-out quadratic smoke result

On 15 November 2026, the 100-step demo checkpoint was evaluated on three held-out four-dimensional quadratic instances (seed 42, budget 100). Median final objective was 13.0525 for the trained policy, 195.745 for random policy weights, and 2.14e-12 for SciPy L-BFGS-B with analytic JAX gradients. Median end-to-end times were 0.635 s, 0.652 s, and 0.0329 s respectively on the onboarding CPU environment. Compilation was included.

The trained policy improved on random initialization but lost decisively to the conventional baseline. This rejects H2 for this small task distribution and checkpoint. Three instances are a smoke check, not statistical evidence for H1. Raw measurements are in `results/heldout-smoke.json`; training loss fell from 0.260531 to 0.00994861 in 100 updates (about 12 seconds).

