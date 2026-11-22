# Release engineering plan: tags, releases, PyPI, versioned docs

Status: **draft for approval (Phase 1, revision 2).** Revision 2 adds your answers to
D1–D5 and the citation and archiving metadata (item 15). Nothing in this plan has been
implemented.
The only Phase 2 work done so far is on a local stash, not committed: the
compiled-engine default flip, with the full suite passing (196 passed, 1 skipped). The
flip is described in item 14.

Phases:
- **Phase 1:** this plan.
- **Phase 2:** implementation on `release/engineering`, then a PR. It is not merged.
- **Phase 3:** after the merge and the manual steps, tags, backfill and the first PyPI
  release, asking before every push.

## 1. Tagging

### Verified mapping

I checked each commit's `pyproject.toml` version, its `CHANGELOG.md` headings and the
merged PR (`gh pr list`). The mapping is correct.

| Tag | Commit | PR | PR title | pyproject at commit | CHANGELOG top heading | Merged (UTC, GitHub) | Merge commit date (+05:30) |
|---|---|---|---|---|---|---|---|
| v0.1.0 | 7e095b5 | #1 | Build learned optimizer with native certification | 0.1.0 | 0.1.0 | 2026-11-14 19:18 | 2026-11-15 00:48 |
| v0.2.0 | 0c68bbe | #2 | Develop outcome-driven NeuralOpt v0.2 research foundation | 0.2.0 | 0.2.0 | 2026-11-15 13:46 | 2026-11-15 19:16 |
| v0.3.0 | ef25291 | #3 | Add NeuralOpt v0.3 stochastic geometry foundations | 0.3.0 | 0.3.0 | 2026-11-15 14:39 | 2026-11-15 20:09 |
| v0.4.0 | c82016f | #4 | Train distribution-aware stochastic geometry for NeuralOpt v0.4 | 0.4.0 | 0.4.0 | 2026-11-15 19:40 | 2026-11-16 01:10 |
| v0.5.0 | 90623b7 | #5 | NeuralOpt v0.5: evaluate transferable learned geometry | 0.5.0 | 0.5.0 | 2026-11-20 21:20 | 2026-11-21 02:50 |
| v0.6.0 | bef56bc | #6 | NeuralOpt v0.6: pre-registered trajectory-curvature study | **0.5.0** | 0.6.0 | 2026-11-21 22:59 | 2026-11-22 04:29 |
| v0.7.0 | new release commit on top of eec9333 | #7 (6e673be), #8 (eec9333), and the Phase 2 PR | — | 0.7.0 after Phase 2 | 0.7.0 | — | — |

### Disagreements and notes

1. **Dates depend on the time zone.**
   - The dates you supplied are the merge commits' local dates (+05:30).
   - GitHub shows merge times in UTC, and on that basis four versions merged one day
     earlier:
     - v0.1.0: 11-14, not 11-15;
     - v0.4.0: 11-15, not 11-16;
     - v0.5.0: 11-20, not 11-21;
     - v0.6.0: 11-21, not 11-22.
   - **D1, decided: UTC dates everywhere** (GitHub's merge dates). These go in the
     CHANGELOG, the release notes and the tag messages. Each tag message also gives the
     +05:30 local merge time.
   - The dates are therefore:
     - v0.1.0: 2026-11-14;
     - v0.2.0: 2026-11-15;
     - v0.3.0: 2026-11-15;
     - v0.4.0: 2026-11-15;
     - v0.5.0: 2026-11-20;
     - v0.6.0: 2026-11-21.
2. **The version number at the v0.6.0 commit is wrong.** At bef56bc, `pyproject.toml`,
   `__init__.__version__` and `CITATION.cff` all say 0.5.0.
   - The tag is still correct: the CHANGELOG and the PR both describe v0.6.
   - However, a wheel built from it would be named `neuralopt-0.5.0`. See D2 in item 6.
3. **Every old tag carries a `zensical.toml`.** That makes the docs backfill (item 7)
   feasible for all six old versions.
4. **Other notes.**
   - `.bumpversion.toml` is stale at 0.4.0, and `__init__.py` hard-codes
     `__version__`. Both are handled in item 5.
   - v0.1.0, v0.2.0 and v0.3.0 were merged from the same branch
     (`feature/research-grade-neuralopt`). That is consistent with the mapping.

### Tag message (annotated)

```
NeuralOpt v0.N.0 (retroactive tag)

Created retroactively on <tag date>. This version was merged to main on
<YYYY-MM-DD> (UTC; <HH:MM> UTC, <local date HH:MM> +05:30) via PR #<n>:
<PR title>.
Release notes: docs/release-notes/v0.N.0.md (on main).
```

- The v0.6.0 message adds: *"pyproject.toml at this commit still reads 0.5.0; see the
  release notes."*
- v0.7.0 is a normal annotated tag on the merge commit of the Phase 2 PR. It is created
  only after you approve.

## 2. Release notes (`docs/release-notes/`)

- `vX.Y.Z.md` for 0.1.0 through 0.7.0, plus an `index.md` table with one row per
  version: version, date, one-line summary, and result type (infrastructure / positive /
  **negative**).
- **Sources:** CHANGELOG entries, the version's research reports (mapped below) and the
  PR commit lists. Reports are linked, not rewritten.
- **Sections:** Highlights, Features, Research results, Fixes, Documentation, Build and
  CI, Breaking changes, and Known limitations. Empty sections are written as "None."
- **Research results** state the outcome in the first sentence. For example: *"Negative:
  learned geometry did not beat the validation-tuned fixed diagonal."*

| Version | Research reports first added in that version |
|---|---|
| 0.1 | theory, experiment-plan, literature, novelty, limitations, benchmarks/smoke |
| 0.2 | training, benchmarks, certification, generalization, trajectory-diagnostics, hamiltonian-{theory,correctness,related-work} |
| 0.3 | langevin-theory, fokker-planck, gibbs-measures, function-spaces, stochastic-{novelty,limitations} |
| 0.4 | distribution-aware-training (negative: learned BAOAB geometry does not beat fixed diagonal) |
| 0.5 | transferable-geometry (negative: learned geometry does not beat tuned fixed geometry) |
| 0.6 | trajectory-curvature-v0.6, -experiments, -literature (negative main result: BB beats every learned arm) |
| 0.7 | engine-speed-v0.7 (speed only), gram-space-v0.7 (pre-registered negative: H1 fails) |

**Breaking changes in 0.7.0:**
- `preconditioned_engine` now defaults to `"compiled"`. Callers that use
  `candidate_method="preconditioned"` with a host/NumPy `gradient=` now get a
  `ValueError` that names `preconditioned_engine="python"`. Before, they ran on the
  Python loop silently.
- `__version__` now comes from the package metadata.

## 3. CHANGELOG.md

- One section per version, in the form `## [0.N.0] - YYYY-MM-DD`, with UTC merge dates
  (D1).
- An `## [Unreleased]` section at the top.
- Each section is a short version of the curated release notes, with bullets kept
  consistent with them. It ends with a link to `docs/release-notes/v0.N.0.md`.
- Comparison links go at the bottom, for example
  `[0.2.0]: https://github.com/bibymaths/neuralopt/compare/v0.1.0...v0.2.0`.

## 4. git-cliff and conventional commits

- **`cliff.toml`** will be replaced with a copy of the globax configuration:
  - `conventional_commits = true` and `filter_unconventional = false`;
  - the same emoji groups and commit preprocessors;
  - `tag_pattern = "v[0-9].*"`;
  - links rewritten to `bibymaths/neuralopt`.
- **git-cliff only drafts.** A new `scripts/draft_release_notes.sh vX.Y.Z` runs
  `git cliff --unreleased --tag vX.Y.Z` and writes to stdout or a scratch file. It never
  writes `CHANGELOG.md` or a `docs/release-notes/` file. Unlike globax, there is no
  changelog workflow that rewrites `CHANGELOG.md`.
- **Release procedure** (in `CONTRIBUTING.md`):
  1. Draft the notes with git-cliff.
  2. Curate them into `docs/release-notes/vX.Y.Z.md`.
  3. Prepend a CHANGELOG section.
  4. Bump the version.
  5. Open a PR, then merge it.
  6. Tag the merge commit.
- **`CONTRIBUTING.md`** gains:
  - the commit types (`feat fix docs perf refactor test build ci chore revert`, plus
    `research` → grouped under "Research", a NeuralOpt addition);
  - the scope convention;
  - a `!` / `BREAKING CHANGE:` footer for breaking changes;
  - the release procedure.
- **PR-title check:** `.github/workflows/pr-title.yml`, using
  `amannn/action-semantic-pull-request` pinned to a release tag, with
  `pull-requests: read` only. It runs on `pull_request` (not `pull_request_target`), so
  fork code never gets elevated permissions.
- **History is not rewritten.** Old commits stay unconventional; git-cliff groups them
  under "Other".
- **D3, decided: squash-merge with the PR title as the commit message.** You set it in
  Settings. With that setting, the PR-title check guarantees conventional commits on
  main. `CONTRIBUTING.md` documents this.

## 5. Version metadata: a single source

- **`pyproject.toml` is the source of truth** (`version = "0.7.0"`).
  - `neuralopt.__version__ = importlib.metadata.version("neuralopt")`.
  - `.bumpversion.toml` is deleted: it is stale, and nothing uses it.
  - `CITATION.cff` (`version` and `date-released`) is set in the release PR, as step 4
    of the release procedure (item 4).
  - `.zenodo.json` **omits `version`**, so Zenodo takes it from the release tag. That
    leaves one less field to keep in sync.
- **`scripts/check_release.py vX.Y.Z`** verifies that:
  - the tag, `pyproject.toml`'s version and `CITATION.cff`'s `version` are all equal;
  - if `.zenodo.json` ever gains a `version`, it must equal the tag too;
  - `CITATION.cff`'s `date-released` equals the date of this version's CHANGELOG
    heading;
  - `CITATION.cff` passes `cffconvert --validate`, run in CI as `uvx
    cffconvert==2.0.0 --validate`. It stays out of the project's dependencies, because
    it pins old `jsonschema` versions;
  - `.zenodo.json` is valid JSON with the required keys: `title`, `upload_type`,
    `description`, `creators`, `license`, `access_right` and `related_identifiers`;
  - `docs/release-notes/vX.Y.Z.md` exists;
  - `CHANGELOG.md` has a `## [X.Y.Z] - ` heading.
- **The metadata checks also run in `ci.yml`, on every PR:** `cffconvert --validate`,
  plus `check_release.py --metadata-only` for the JSON and the version agreement.
  - With `--backfill`, it skips the three-way version check (D2) and checks only that the
    notes and the CHANGELOG entry exist **on main**.
- **Tests:**
  - `__version__` matches `pyproject.toml`;
  - the release checker passes on 0.7.0;
  - the release checker fails when a version mismatches, a notes file is missing, the
    `date-released` disagrees with the CHANGELOG, or `.zenodo.json` is invalid.

## 6. `.github/workflows/release.yml`

**Triggers:**
- `push: tags: ['v*']`;
- `workflow_dispatch` with inputs `tag` (string) and `publish_pypi` (boolean, default
  false), used for backfill.

**Workflow-level settings:**
- `permissions: contents: read`;
- `concurrency: release-${{ tag }}` with `cancel-in-progress: false`.

**Jobs:**

1. **`verify`** checks out `ref: <tag>` and runs `check_release.py`. For tags below
   v0.7.0, it runs `--backfill` against main's files.
   - Outputs: `version`, `is_backfill`, and `pypi_ok`. `pypi_ok` is true when
     `Version(tag) >= 0.7.0` and either the run was a tag push or `publish_pypi` is
     true. The comparison uses `packaging.version`.
2. **`test`** runs the full quality gate at the tag: `uv sync --extra dev --extra docs`,
   `ruff check .`, `pyright`, `pytest -q`, `zensical build --strict`, `uv build`, and
   `twine check dist/*`.
   - It is **skipped for backfill**, because those commits passed CI in their own PRs
     and old tags lack the current tooling. The release notes say so.
3. **`build`** runs `uv build` at the tag and uploads `dist/` with
   `actions/upload-artifact` and `retention-days: 3`.
   - **D2, decided: v0.6.0 gets no build files.** Its sdist and wheel would be named
     0.5.0, so `build` is skipped for v0.6.0. Its release notes explain why. All other
     versions get both an sdist and a wheel.
4. **`pypi`** has `needs: [verify, test, build]` and `if: pypi_ok`.
   - `environment: pypi`, and `permissions: id-token: write` in this job only.
   - It uses `pypa/gh-action-pypi-publish` pinned to a release tag. That is trusted
     publishing, with no tokens.
5. **`github-release`** has `permissions: contents: write`.
   - It uses `gh release create vX.Y.Z --title "NeuralOpt vX.Y.Z" --notes-file
     docs/release-notes/vX.Y.Z.md --prerelease --verify-tag`, then attaches the dist
     files.
   - The notes file is read from main for backfill. All 0.x releases are marked as
     pre-releases.
   - For backfill it waits for nothing on PyPI. For v0.7.0 it runs after `pypi`
     succeeds.
6. **`docs`** calls the reusable deploy steps from item 7 for the tag's minor version.
   - It runs in `concurrency: gh-pages` (shared with `docs.yml`, `cancel-in-progress:
     false`), so deploys never overlap.

**Common to all jobs:**
- Actions are pinned to full release tags: `actions/checkout@v4.x.y`,
  `astral-sh/setup-uv@v6.x.y`, and so on, with exact values resolved in Phase 2.
- **Option:** pin to commit SHAs instead. That is safer but noisier, so say if you want
  it.
- No other workflow uploads artifacts. Release files live only on the GitHub release.

## 7. Versioned docs (Zensical plus its mike fork)

**Tooling:**
- Zensical's versioning is a temporary fork of mike 2.2.0 (`squidfunk/mike`). It is not
  on PyPI.
- It is installed from git, pinned to a commit hash: `mike @
  git+https://github.com/squidfunk/mike@2d4ad799442f4592db8ad53b179bfb33db8c69ac`.
  - That is the head of `master` on 2026-04-17. The fork of `jimporter/mike` was last
    pushed on 2026-10-11.
  - It is used in the `docs` extra and in the workflows.
- The `zensical.toml` setting is `[project.extra.version] provider = "mike"`, with
  `alias = true` and `default = "latest"`.
- **What I read in the fork at that commit:**
  - **Zensical version:** it requires `zensical >= 0.0.30`. The installed 0.1.1
    satisfies that, so **Zensical is not upgraded.** If Phase 2 nonetheless shows an
    incompatibility, Zensical is upgraded and `zensical build --strict` is run again.
  - **How it builds:** `mike deploy` runs `zensical build --clean`, **without**
    `--strict`. Strictness is therefore a separate `zensical build --strict` step,
    run before every deploy.
  - **Config file:** `mike` reads `zensical.toml`, or the file given with `-F/--config-file`.
    It resolves `site_dir` relative to that file's directory.
  - **Commands:** mike 2.2.0's (`deploy --push --update-aliases`, `set-default`, `list`,
    `-b gh-pages`).
- The Zensical versioning page (`zensical.org/docs/setup/versioning/`) currently
  redirects to a 404. The fork's README points to the upstream mike documentation.

**Layout on `gh-pages`:**
- one folder per minor version, `0.1/` … `0.7/`, each titled with its full version (for
  example "0.7.0");
- aliases `latest` (the newest release) and `dev` (main);
- `mike set-default latest`;
- `site_url = https://bibymaths.github.io/neuralopt/`.

**`.github/workflows/docs.yml`:**
- On PRs it runs `zensical build --strict` only, with no deploy.
- On pushes to main it runs `mike deploy --push --update-aliases dev`.
- It runs in concurrency group `gh-pages`, with `permissions: contents: write` only in
  the deploy job.

**Backfill (`scripts/backfill_docs.py`):** for each old tag it:
1. runs `git worktree add <tmp> vX.Y.0`;
2. writes an **untracked overlay config** at the worktree root: the tag's
   `zensical.toml` plus `site_url` and the version provider. It is named
   `.zensical-versioned.toml` and passed with `-F`. It has to sit at the root, because
   `docs_dir` and `site_dir` resolve relative to the config file. It is never committed,
   and the worktree is deleted afterwards.
3. runs `mike deploy --config-file <overlay> 0.N` with `--title 0.N.0`;
4. runs `zensical build --strict -f <overlay>` first. If that fails, it records the
   failure and deploys anyway, since mike's own build is non-strict;
5. removes the worktree.

The script:
- writes a report to `artifacts/docs-backfill-report.json` (gitignored scratch, not a
  workflow artifact) listing strict, non-strict and failed builds;
- defaults to `--no-push` (a local `gh-pages` commit only). Pushing needs `--push` and
  happens only in Phase 3, after you approve.

**Phase 2 local check:** build two versions (0.6 and 0.7) into a local `gh-pages` with
`mike deploy` (no push), then `mike serve` and check the version selector. The result
is reported.

## 8. Docs site structure (current version)

The `zensical.toml` nav will be (new pages marked ★, existing pages reused):

- **Home** (`index.md`, rewritten to a landing page ★)
- **Getting started:** Installation ★, Quickstart ★, When to use NeuralOpt ★
- **User guide:** Learned mode ★, Certified mode ★ (built from `research/certification.md`
  usage parts plus the README), Compiled engine ★ (the usage parts of
  `research/engine-speed-v0.7.md`; the report itself is linked, not moved)
- **Research reports by version:** 0.1 … 0.7 groups pointing at the existing
  `docs/research/*.md` files, with no rewrites
- **Release notes:** `release-notes/index.md` and each version
- **Changelog:** `changelog.md` ★, which includes the root `CHANGELOG.md` via the
  `pymdownx.snippets` include
- **Limitations:** `research/limitations.md` plus `research/stochastic-limitations.md`
- **API reference** ★: `mkdocstrings` for `neuralopt.__all__` (as in globax;
  `mkdocstrings-python` is added to the `docs` group)
- **Citing NeuralOpt** ★ (`citing.md`):
  - the `CITATION.cff` details: title, version, author with ORCID, license, repository;
  - "cite the exact version or commit";
  - a BibTeX `@software` entry generated by `cffconvert -f bibtex`;
  - (later) the concept DOI.
  - A test checks that the page's version, title and ORCID match `CITATION.cff`.
- `docs/superpowers/` and `docs/plans/` stay out of the nav. They are internal working
  documents, built but not listed.

**"When to use NeuralOpt" says it plainly:**
- For solving problems, use SciPy L-BFGS-B or Optimistix LBFGS. On the v0.7 equal-work
  benchmark they reach 1e-6 on 32/32 tasks, where learned NeuralOpt geometry reaches it
  on 0/32.
- NeuralOpt is for research on learned optimizers, and for its restricted certified mode
  on the supported expression subset.

## 9. README

The README will contain, in order:
- the logo;
- a one-line description;
- badges: CI, Docs, PyPI version, Python versions, License MIT, Ruff, uv (downloads only
  after the first PyPI release);
- **"Status: alpha research package; learned components do not outperform classical
  optimizers in our benchmarks (see release notes)."**;
- install (`pip install neuralopt` once on PyPI; `uv sync` for development);
- the quickstart (the learned and certified examples, both checked by a doctest-style
  test);
- when to use NeuralOpt;
- links to docs, release notes and the changelog;
- citation: a short BibTeX block and a link to the **Citing NeuralOpt** docs page and
  `CITATION.cff` (a DOI badge after the first Zenodo archive);
- license.

## 10. Logo

- **Original SVG, written by hand, in the globax palette:**
  - deep navy `#0B1B3F` for the wordmark;
  - gradient cyan `#22D3EE` → blue `#2563EB` → purple `#8B5CF6`.
- **Icon:** a contour basin, three or four nested elliptical level sets, with a short
  chain of connected nodes (a descent path) running from the rim into the minimum. The
  final node is highlighted. There is no globe and no cubes.
- **Wordmark:** lowercase geometric "neuralopt" in navy, with "opt" in the gradient,
  mirroring globax's accented "x". Letterforms are drawn as paths, so no font is needed
  at render time.
- **Files in `docs/assets/images/`:**
  - `logo.svg` (source) and `logo.png` (1448×1086, matching the globax canvas);
  - `logo-dark.svg` and `logo-dark.png`, with a light wordmark for dark backgrounds;
  - `favicon/`: `favicon.ico`, 16/32 PNGs, `apple-touch-icon.png`, android
    192/512 and `site.webmanifest`, made from the icon alone.
- PNGs are rendered with `rsvg-convert` or `cairosvg`, whichever is available, and the
  command used is recorded.
- **Usage:**
  - the README uses `<picture>` with light and dark sources;
  - `zensical.toml` sets `favicon` and `theme.logo`;
  - the palette is navy/indigo plus deep-purple accent, as in globax.
- You review the rendered PNGs before the PR.

## 11. Manual steps for you (I cannot do these)

1. ✅ Check that `neuralopt` is free on PyPI (it is now reserved by your pending
   publisher).
2. ✅ The PyPI pending trusted publisher is created. **Check it matches exactly:**

   | Field | Value |
   |---|---|
   | PyPI project name | `neuralopt` |
   | Owner | `bibymaths` |
   | Repository name | `neuralopt` |
   | Workflow name | `release.yml` (the file name only, no path) |
   | Environment name | `pypi` |

   - The plan uses exactly `.github/workflows/release.yml`, and a `pypi` job with
     `environment: pypi`.
   - The publish step runs directly in that file, not in a reusable workflow that it
     calls. Otherwise PyPI's OIDC claim would name the other file.
   - A pending publisher does not reserve the name. If someone else registers
     `neuralopt` before v0.7.0 is published, it lapses.
3. Create the GitHub environment **`pypi`** (Settings → Environments), with yourself as
   required reviewer. Optionally restrict deployment to tags `v*`.
4. Set **Pages** to "Deploy from a branch", branch `gh-pages`, folder `/`, after the
   branch exists. The branch is created by the first `mike` push in Phase 3.
5. ✅ D4: the repository is public (`gh repo view` confirms it).
6. D3: in Settings → General, allow squash merging and set the default commit message
   to "Pull request title". You said you will do this.
7. **Zenodo, after the backfill only.** Enable the Zenodo GitHub integration for
   `bibymaths/neuralopt` only **after** the v0.1.0–v0.6.0 GitHub releases exist and
   **before** v0.7.0 is tagged.
   - Zenodo archives on "release published" events, so v0.7.0 becomes the first archived
     version.
   - Phase 3 has a hard stop between the backfill and the v0.7.0 tag for this step.
8. **After the first archive,** send me the concept DOI. A follow-up PR (`docs:` or
   `chore:`) adds it to:
   - `CITATION.cff` (`doi:` and `identifiers`);
   - `.zenodo.json` `related_identifiers`;
   - a DOI badge in the README and the Citing page.
9. Optional: set the repository's website field to
   `https://bibymaths.github.io/neuralopt/`, so it shows on GitHub.

## 12. Scope

- v0.1.0–v0.6.0 get annotated tags, GitHub pre-releases (with assets except v0.6.0,
  per D2) and versioned docs.
- They are **never published to PyPI**: `pypi_ok` is false below 0.7.0, and the backfill
  runs with `publish_pypi=false`.
- v0.7.0 is the first PyPI release.

## 13. Actions storage

- Artifacts are kept only between the jobs of one release run, with `retention-days: 3`.
- CI and docs workflows upload no artifacts. Docs deploy by pushing `gh-pages`, not by
  Pages artifacts.
- Release files are kept on the GitHub release.

## 14. Also in Phase 2: the compiled-engine default flip

This is already implemented test-first on a local stash, and is still uncommitted:

- **Default changed:** `minimize(..., preconditioned_engine=...)` now defaults to
  `"compiled"`.
- **v0.6 pinned:** `scripts/run_trajectory_curvature_v0_6.py` passes
  `preconditioned_engine="python"` explicitly at both preconditioned call sites. A new
  test, which parses the script's syntax tree, enforces this.
- **Script hash recorded, results untouched:** the script's sha256 changes from
  `35c1e401…50d1`. The new hash goes in the release notes and `engine-speed-v0.7.md`.
  `results/trajectory-curvature-v0.6.0.json` is not touched.
- **Host-gradient tests pinned:** three tests in `tests/test_trajectory.py` use host
  gradients that are deliberately inconsistent with the objective. They test the Python
  loop, so they are now pinned to `"python"`.
- **New tests:** the default is compiled; and with a host gradient, the default fails
  closed with a message naming `preconditioned_engine='python'`.
- **v0.7 scripts:** they all pass `"compiled"` explicitly, so they are unaffected.
- **Measured:** full `pytest` gives 196 passed and 1 skipped (`diffrax` not installed).
  The legacy golden tests pass.

**D5, decided:** this PR adds a dated note (2026-11-22) to `gram-space-v0.7.md` →
"Decision and next work". It says the user chose **amortized optimization over problem
families** as the next direction, and that learned guidance for certified search is
**deferred, not cancelled**. The note is added below the existing bullet, which stays as
written for the record.

## 15. Citation and archiving metadata

### `CITATION.cff` (updated)

```yaml
cff-version: 1.2.0
message: >-
  If you use NeuralOpt, please cite the exact version or commit you used.
  NeuralOpt is an alpha research package; it makes no performance claims,
  and several of its research results are negative (see the release notes).
type: software
title: "NeuralOpt: a compact learned optimizer in JAX with restricted, replayable interval certification"
version: 0.7.0
date-released: <set in the release PR>
authors:
  - family-names: Mishra
    given-names: Abhinav
    affiliation: Independent Researcher
    email: mishraabhinav36@gmail.com
    orcid: "https://orcid.org/0009-0005-3179-7408"
repository-code: "https://github.com/bibymaths/neuralopt"
url: "https://bibymaths.github.io/neuralopt/"
license: MIT
keywords: [learned optimization, learned optimizers, meta-learning, JAX, transformer,
           preconditioning, quasi-Newton, interval arithmetic, global optimization,
           certified optimization, Langevin dynamics, Hamiltonian Monte Carlo]
abstract: >-
  <three or four plain sentences: what the package does (a transformer-proposed update
  rule in JAX; an opt-in compiled loop; a separate, model-free interval certification mode for a
  restricted expression subset), and its honest status: alpha; across v0.4-v0.7 the
  learned geometries did not beat validation-tuned fixed or Barzilai-Borwein baselines,
  and classical L-BFGS outperforms it on the reported benchmarks.>
```

- The email is the one you supplied. A public `CITATION.cff` publishes it; say if you'd
  rather leave it out.
- The `title` one-liner also becomes `pyproject.toml`'s `description` and the README's
  first line, so all three agree.

### `.zenodo.json` (updated; the file exists from v0.1.0 with minimal fields)

The file follows globax's format:
- `title`: the same as `CITATION.cff`'s;
- `upload_type`: `"software"`;
- `description`: an honest summary, including the **negative-results status**;
- `creators`: `[{"name": "Mishra, Abhinav", "affiliation": "Independent Researcher",
  "orcid": "0009-0005-3179-7408"}]`;
- `license`: `"MIT"`;
- `access_right`: `"open"`;
- `language`: `"eng"`;
- `keywords`: as in `CITATION.cff`;
- `related_identifiers`:
  - `https://github.com/bibymaths/neuralopt`, `isSupplementTo`, `software`;
  - `https://bibymaths.github.io/neuralopt/`, `isDocumentedBy`, `other`;
- **no `version` field**, as noted in item 5.

### Zenodo timing and the DOI follow-up

- Zenodo is enabled only after the backfill (manual step 7), so v0.7.0 is the first
  archived version.
- After the first archive, a follow-up PR adds the concept DOI (manual step 8). That
  PR's tests are extended so the DOI in `CITATION.cff`, `.zenodo.json`, the README badge
  and `citing.md` must agree.

## Phase 3 order (each push needs your go-ahead)

1. Push the annotated tags v0.1.0–v0.6.0. **Ask first.**
2. Backfill the GitHub releases: `release.yml` via `workflow_dispatch`, one tag at a
   time, with `publish_pypi=false`. v0.6.0 gets no files.
3. Backfill the docs: `scripts/backfill_docs.py --push` (0.1–0.6). Then deploy `dev`
   from main, and report the strict/non-strict list. **Ask before pushing `gh-pages`.**
4. **STOP:** you set Pages to the `gh-pages` branch and enable Zenodo (manual steps 4
   and 7).
5. Tag v0.7.0 on main. **Ask first.** `release.yml` then does the following:
   - verify, test and build;
   - publish to PyPI (after your approval in the `pypi` environment);
   - create the GitHub release;
   - deploy docs `0.7` with aliases `latest` and default.
   - Zenodo archives the release.
6. Report the release URLs, the docs URL and the PyPI page, then open the DOI follow-up
   PR once you send me the DOI.

## Risk register

| Risk | Mitigation |
|---|---|
| R1: the mike fork is incompatible with the installed Zensical | Confirm in Phase 2; bump Zensical; record the version |
| R2: old tags fail `--strict` with the current Zensical | Build non-strict, list them in the backfill report |
| R3: a private repository blocks Pages and public badges | Resolved: the repository is public |
| R4: a release workflow bug publishes the wrong version | `check_release.py` gate; PyPI only at ≥0.7.0 and behind the `pypi` environment reviewer |
| R5: the engine default flip breaks host-gradient users | It fails closed with an actionable message; listed under Breaking changes |
| R6: frozen artifacts are touched | No writes under `results/` or `checkpoints/`; the PR diff is checked for those paths |

## Decisions (answered 2026-11-22)

- **D1:** UTC (GitHub) merge dates everywhere; the +05:30 time goes in the tag messages.
- **D2:** no files for v0.6.0, explained in its notes.
- **D3:** squash-merge with the PR title (you will set this).
- **D4:** the repository is public.
- **D5:** include the amortized-direction note.

| Risk | Mitigation |
|---|---|
| R7: Zenodo is enabled too early and archives the backfilled 0.x releases | Manual step 7; Phase 3 stops before tagging v0.7.0 |
| R8: the citation metadata drifts from the version | `check_release.py` cross-checks every version field; `cffconvert --validate` in CI |
