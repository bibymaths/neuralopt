# Contributing

Open an issue describing the scientific question or defect before a large change. Keep inference independent of conventional solver backends, add a falsifiable test for numerical claims, preserve deterministic seeds, and label heuristic and rigorous results separately. Run `uv sync --extra dev`, `uv run ruff check .`, and `uv run pytest` before submitting a pull request. New certification primitives require containment tests and a documented rounding argument.

