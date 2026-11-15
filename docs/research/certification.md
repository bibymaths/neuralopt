# Certified scheduling

Certification now exposes deterministic lower-bound or width region priority,
widest or round-robin legal splitting, and callbacks for learned ranking and
split selection. Callbacks affect scheduling only: interval lower bounds,
incumbents, pruning, and termination remain native and rigorous under the stated
floating-point assumptions.
Default lower-bound scheduling reads the global bound from the scheduling heap
root. Alternative and learned priorities maintain a second lower-bound heap
with lazy deletion, avoiding a full active-set scan after every processed node.

Every terminal pruned box is serialized. Independent replay re-evaluates each
interval bound, checks that boxes lie inside the original domain, compares total
leaf and domain volume, rejects positive-volume overlap, and checks the final
gap. Partition replay follows shared bisection boundaries recursively, avoiding
quadratic all-pairs overlap checks. This closes the v0.1.0 verifier's
domain-accounting omission while remaining practical for thousands of leaves.

On the dependency-stress expression `x[0] - x[0]`, 8,191 processed nodes and
8,192 retained leaves took about 0.89 seconds for search and 0.68 seconds for
replay after the current fixes. Scaling from 2,048 to 16,384 leaves was close to
linear. Search time is dominated by interval-expression evaluation and `Interval`
object validation; replay is dominated by recursive partition reconstruction,
followed by re-evaluating every retained interval bound.

No learned certification policy has yet been trained or benchmarked. Therefore
there is no supported node-count or wall-time speedup claim.
