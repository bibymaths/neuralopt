# Model artifact provenance

`demo-v0.1.0.npz` was generated on 15 November 2026 with `neuralopt train-demo --steps 100 --seed 42`. It used the deterministic 14-instance training partition from `task_split(42, 20, 4)`, Adam at the CLI defaults, gradient clipping at 1.0, and float32 JAX CPU execution. Loss changed from 0.260531 to 0.00994861 in approximately 12 seconds. The same bytes are packaged under `src/neuralopt/checkpoints/` for the default API.

This artifact has not demonstrated broad generalization. It must not be described as a universal optimizer.

