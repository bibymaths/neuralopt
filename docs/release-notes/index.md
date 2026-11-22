# Release notes

Every NeuralOpt release is an alpha research release. Versions before 0.7.0 were tagged
retroactively on 2026-11-22 and are not on PyPI.

| Version | Date (UTC) | Summary | Result |
|---|---|---|---|
| [0.7.0](v0.7.0.md) | 2026-11-22 | Compiled engine (speed only); pre-registered Gram-space history model does not beat the untrained BB step; first PyPI release. | negative |
| [0.6.0](v0.6.0.md) | 2026-11-21 | Secant-history features and a shared preconditioned loop; an untrained BB step beats every learned arm by about one decade. | negative |
| [0.5.0](v0.5.0.md) | 2026-11-20 | Scale-aware bounded geometry; learned geometry does not beat validation-tuned fixed geometry. | negative |
| [0.4.0](v0.4.0.md) | 2026-11-15 | Trained stochastic mobility and mass geometry; learned BAOAB geometry does not beat a fixed diagonal. | negative |
| [0.3.0](v0.3.0.md) | 2026-11-15 | Langevin and BAOAB candidates, Fokker–Planck diagnostics and a function-space reference. | infrastructure |
| [0.2.0](v0.2.0.md) | 2026-11-15 | Outcome-driven training, trajectory diagnostics, certified scheduling and Hamiltonian proposals. | infrastructure |
| [0.1.0](v0.1.0.md) | 2026-11-14 | Transformer optimization policy with restricted, independently replayed interval certificates. | infrastructure |

For a condensed list of changes, see the [changelog](../changelog.md).
