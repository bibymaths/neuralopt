# Trajectory diagnostics

The v0.2 diagnostic mode records one row per objective evaluation and does not
perform extra objective calls. It distinguishes raw, preconditioned, projected,
and accepted directions; objective and incumbent state; projection; boundary
saturation; and exact small-dimensional preconditioner eigenvalues.

## Reproduction and dominant failure

`uv run python scripts/run_research_smoke.py` reproduces the published trained
median on the original held-out split (13.0525). The random-weight value is not
reproducible because v0.1.0 did not record its seed. With seed 123 its median is
27.4117 rather than 195.745. The original files and checkpoint remain unchanged.

On six single-instance, four-dimensional smoke tasks (49 proposals after the
initial evaluation), the trained raw direction has median descent cosine 0.833
on the isotropic quadratic and 0.994--0.999 on the other five tasks.
Preconditioning changes this little and has median condition number 1.1--1.4.
Nevertheless, acceptance is 0% on four tasks and 4.1% on the other two. The
independent proposal head consumes 24/49 evaluations and accepts none.

This evidence rejects bad direction and ill-conditioned preconditioning as the
dominant explanation in this sample. The primary failures are excessive step
length, an unsupervised proposal head, and poor budget allocation. A separate
implementation defect compounded this: rejection moved `x` without evaluating
the new point, leaving its stored value inconsistent with its gradient. v0.2
leaves the current state unchanged after rejection.

The full machine-readable record is `results/research-smoke-v0.2.0.json`. This
is a diagnostic smoke study, not an independent statistical benchmark.
