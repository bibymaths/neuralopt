# Certified scheduling

Certification now exposes deterministic lower-bound or width region priority,
widest or round-robin legal splitting, and callbacks for learned ranking and
split selection. Callbacks affect scheduling only: interval lower bounds,
incumbents, pruning, and termination remain native and rigorous under the stated
floating-point assumptions.

Every terminal pruned box is serialized. Independent replay re-evaluates each
interval bound, checks that boxes lie inside the original domain, compares total
leaf and domain volume, rejects positive-volume overlap, and checks the final
gap. This closes the v0.1.0 verifier's domain-accounting omission.

No learned certification policy has yet been trained or benchmarked. Therefore
there is no supported node-count or wall-time speedup claim.
