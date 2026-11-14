# Mathematical specification

At step `t`, normalized coordinate tokens contain the current point, incumbent, scaled gradient, gradient magnitude, and availability mask. Two learned task tokens aggregate the variable-cardinality sequence. Multi-head self-attention produces coordinate states. Heads emit direction `d`, independent proposal `q`, step `alpha`, positive diagonal `D = softplus(z)+1e-4`, and low-rank `U`. The update is

\[x_{t+1}=\Pi_{[l,u]}(x_t+\alpha(Dd+U(U^Td))).\]

The implementation never forms the dense preconditioner. Its strict positive diagonal makes `D + UUᵀ` positive definite, although the learned direction does not guarantee descent. Independently evaluated proposals alternate with update candidates. A fixed reject-and-contract rule prevents propagation of worse or nonfinite candidates.

Certification uses natural interval extensions with one-ULP outward expansion after every supported operation. Branch-and-bound preserves the domain by bisection, evaluates midpoint incumbents with an upward-rounded scalar value, and prunes only if the interval lower bound is no smaller than the feasible upper bound. Termination requires `U - min_R L(R) <= epsilon`. The serialized certificate retains every active leaf; verification re-evaluates those bounds without model output.

Assumptions: IEEE-754 binary64, correctly rounded basic NumPy operations and elementary functions supplied by the platform libm, deterministic expression semantics, compact finite boxes, and no NaNs. Since IEEE 1788 decoration and a formally verified libm are absent, certificates are validated numerical results under these assumptions, not machine-checked formal proofs.

