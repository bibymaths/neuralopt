# Model artifact provenance

`demo-v0.1.0.npz` was generated on 15 November 2026 with `neuralopt train-demo --steps 100 --seed 42`. It used the deterministic 14-instance training partition from `task_split(42, 20, 4)`, Adam at the CLI defaults, gradient clipping at 1.0, and float32 JAX CPU execution. Loss changed from 0.260531 to 0.00994861 in approximately 12 seconds. The same bytes are packaged under `src/neuralopt/checkpoints/` for the default API.

This artifact has not demonstrated broad generalization. It must not be described as a universal optimizer.

## v0.6 trajectory-curvature checkpoints

`trajectory-v0.6.0-<arm>-seed<s>.npz` (21 files: `tf-legacy`, `tf-h0/4/8`, `mlp-h0/4/8`;
training seeds 0, 1, 2) were trained with
`uv run python scripts/run_trajectory_curvature_v0_6.py`: 400 Adam updates (1e-3, global-norm
clip 1.0) of `objective="history"` over 21 training tasks, float32 JAX CPU. Each file's
`metadata` records arm, seed, model kind, split hashes, generator, RNG and git HEAD;
`results/trajectory-curvature-v0.6.0.json` records each file's sha256. History arms carry
`schema = "neuralopt-policy/v0.6-history-v1"`; `tf-legacy` and the MLP files use the
schema their config implies (`tf-legacy` has no schema key). All were trained for the
`candidate_method="preconditioned"` loop, where `diag + UUᵀ` is an **inverse** metric; do not
use them as HMC masses. None beat validation-tuned fixed geometry under the pre-registered
rule (`docs/research/trajectory-curvature-v0.6.md`); they are research artifacts only.
