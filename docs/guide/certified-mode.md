# Certified mode

`minimize(problem, mode="certified", absolute_gap=..., max_evaluations=...)` proves a
global lower bound by interval branch-and-bound. **No learned component affects the
bounds**: interval evaluation with outward rounding is the only source of the
certificate.

## Supported expressions

Build the objective with `Variable` and the operators `+`, `-`, `*`, square (`**2`),
`sin` and `exp`, and pass it as `certified_expression`. Arbitrary JAX code cannot enter
certified mode; division, logarithms, arbitrary powers and constraints are unsupported
and fail closed.

```python
from neuralopt import Problem, Variable, minimize, verify_certificate

x = Variable()
expr = (x[0] - 0.25) ** 2
p = Problem(lambda z: (z[0] - 0.25) ** 2, (-1.0, 1.0), 1, certified_expression=expr)
r = minimize(p, mode="certified", absolute_gap=1e-4, max_evaluations=10000)
print(r.status, r.lower_bound, r.upper_bound)
assert verify_certificate(r.certificate)
```

## Independent replay

`verify_certificate` re-evaluates every retained interval bound, checks that the leaf
boxes lie inside the domain and cover it without positive-volume overlap, and checks the
final gap. It does not use the search code's state.

## Assumptions

Outward rounding uses `numpy.nextafter` in binary64 and assumes correctly rounded host
operations and libm. It is **not** IEEE 1788 conformant and is not a formal proof across
arbitrary platforms. See [certified scheduling](../research/certification.md) and the
[limitations](../limitations.md).
