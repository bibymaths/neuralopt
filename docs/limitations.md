# Limitations

NeuralOpt is an alpha research package. Read these before interpreting any result.

## Learned optimization

- **No learned geometry has beaten its simple baselines.** Learned BAOAB geometry did
  not beat a fixed diagonal (v0.4); learned geometry did not beat validation-tuned fixed
  geometry (v0.5); an untrained Barzilai–Borwein step beat every learned arm by about a
  decade (v0.6); the pre-registered Gram-space model did not beat BB (v0.7). Classical
  L-BFGS outperforms the learned optimizer on the reported benchmarks.
- The bundled `demo-v0.1.0` checkpoint is a smoke artifact.
- Learned mode needs gradients and supports box constraints only; residuals and
  nonlinear constraints are metadata, not implemented constrained optimization.
- The compiled engine needs JAX-traceable objectives; host gradients run on the Python
  loop only.

## Certification

- Certified mode supports an explicit expression language, not JAXPR or arbitrary JAX.
- The `nextafter` backend assumes correctly rounded binary64 host operations and libm;
  it is not IEEE 1788 conformant and is not a formal proof across platforms.
- Replay checks the retained bounds and domain accounting; it does not prove the
  implementation or hardware.
- Neural or stochastic predictions never certify global optimality.

## Detailed records

- [Limitations as recorded in v0.1–v0.2](research/limitations.md)
- [Stochastic research limitations and gate record (v0.3–v0.4)](research/stochastic-limitations.md)
- Per-version limitations: the "Known limitations" section of each
  [release note](release-notes/index.md).
