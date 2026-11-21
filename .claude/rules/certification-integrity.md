---
paths:
  - "src/neuralopt/certification.py"
  - "tests/**"
  - "docs/research/certification.md"
---
# Certification is not learned

- Interval bounds and outward rounding remain the only source of supported mathematical certificates.
- No neural probabilities, HMC acceptance or FP residuals can remove a region or change a validated lower bound.
- Run the independent certificate replay and fail-closed unsupported-expression tests for any modification affecting candidate injection or result serialization.
- Report assumptions: binary64/libm restrictions and serializable expression subset; don't claim an IEEE 1788 formal proof.
