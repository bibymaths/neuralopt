# Contributing

Open an issue describing the scientific question or defect before a large change. Keep
inference independent of conventional solver backends, add a falsifiable test for
numerical claims, preserve deterministic seeds, and label heuristic and rigorous results
separately. New certification primitives require containment tests and a documented
rounding argument.

## Local checks

```bash
uv sync --locked --extra dev --group docs
uv run ruff check .
uv run pyright
uv run pytest -q
uv run zensical build --strict
uv build --build-constraints build-constraints.txt && uv run --frozen twine check dist/*
uv run python scripts/check_release.py      # release metadata consistency
uvx cffconvert==2.0.0 --validate            # CITATION.cff schema
```

## Commit and pull-request titles

Pull requests are **squash-merged**, and the PR title becomes the commit message on
`main`. PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/);
a CI check (`pr-title.yml`) enforces it.

```
<type>(<optional scope>)!: <imperative summary>
```

| Type | Use for |
|---|---|
| `feat` | a new user-visible capability |
| `fix` | a bug fix |
| `research` | experiments, pre-registrations, results and their reports |
| `docs` | documentation only |
| `perf` | a speed or memory improvement with unchanged results |
| `refactor` | code change with no behaviour change |
| `test` | tests only |
| `build` | packaging, dependencies |
| `ci` | GitHub Actions |
| `chore` | maintenance (`chore(release): vX.Y.Z` is skipped by git-cliff) |
| `revert` | reverting a previous commit |

Scopes are optional and name the area, for example `feat(compiled): ...`,
`research(v0.8): ...` or `docs(release-notes): ...`.

Mark a breaking change with `!` after the type/scope, and explain it in a
`BREAKING CHANGE:` footer of the squash commit body. History before v0.7.0 does not use
this convention and is not rewritten.

## Releasing

`pyproject.toml` is the single source of the version; `neuralopt.__version__` reads the
installed package metadata.

1. **Draft:** `scripts/draft_release_notes.sh vX.Y.Z` prints a git-cliff draft from the
   conventional commits since the last tag. It writes no files.
2. **Curate** the draft into `docs/release-notes/vX.Y.Z.md` (sections: Highlights,
   Features, Research results, Fixes, Documentation, Build and CI, Breaking changes,
   Known limitations; state negative results plainly) and add it to
   `docs/release-notes/index.md` and the `zensical.toml` nav.
3. **Changelog:** move the `[Unreleased]` items into a new
   `## [X.Y.Z] - YYYY-MM-DD` section of `CHANGELOG.md` (UTC date of the release) and add
   the compare link at the bottom.
4. **Bump** `version` in `pyproject.toml`, and `version` and `date-released` in
   `CITATION.cff` (the same date as the CHANGELOG heading). `.zenodo.json` carries no
   version; Zenodo takes it from the tag.
5. **Check:** `uv run python scripts/check_release.py vX.Y.Z` and the full local checks.
6. Open a PR titled `chore(release): vX.Y.Z`; squash-merge it after CI passes.
7. **Tag** the squash commit on `main` with an annotated tag and push it:
   `git tag -a vX.Y.Z -m "NeuralOpt vX.Y.Z" && git push origin vX.Y.Z`.
   The `release.yml` workflow verifies the metadata, runs the quality gate, builds,
   publishes to PyPI (after approval in the `pypi` environment), creates the GitHub
   release from the curated notes and deploys the versioned docs.
