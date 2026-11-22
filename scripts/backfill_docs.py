"""Deploy versioned docs for old (retroactive) tags to gh-pages with Zensical's mike fork.

For each tag: check the tag out in a temporary git worktree, write an *untracked* overlay
config at the worktree root (the tag's zensical.toml plus site_url and the mike version
provider; nothing is committed to the tag), try ``zensical build --strict`` and record the
outcome, then ``mike deploy`` (mike's own build is non-strict) as version ``X.Y`` titled
``X.Y.Z``. The worktree is removed afterwards.

Commits go to the local gh-pages branch. Nothing is pushed without ``--push``.
Run from the repository root with the docs group installed::

    uv run --group docs python scripts/backfill_docs.py            # local only
    uv run --group docs python scripts/backfill_docs.py --push     # Phase 3, after approval
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

SITE_URL = "https://bibymaths.github.io/neuralopt/"
DEFAULT_TAGS = [f"v0.{i}.0" for i in range(1, 7)]
OVERLAY_NAME = ".zensical-versioned.toml"


def minor(tag: str) -> str:
    return ".".join(tag.removeprefix("v").split(".")[:2])


def overlay_config(original: str) -> str:
    """The tag's config plus ``site_url`` and the mike version provider."""
    parsed = tomllib.loads(original)
    project = parsed.get("project", {})
    if "site_url" in project or "version" in project.get("extra", {}):
        raise ValueError("config already sets site_url or extra.version; overlay by hand")
    if not original.lstrip().startswith("[project]"):
        raise ValueError("expected the config to start with a [project] table")
    head, _, rest = original.partition("[project]\n")
    overlay = (
        f'{head}[project]\nsite_url = "{SITE_URL}"\n{rest.rstrip()}\n\n'
        '[project.extra.version]\nprovider = "mike"\ndefault = "latest"\nalias = true\n'
    )
    tomllib.loads(overlay)  # must stay valid TOML
    return overlay


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def backfill(tag: str, repo: Path, push: bool, branch: str) -> dict[str, object]:
    record: dict[str, object] = {"tag": tag, "version": minor(tag)}
    worktree = Path(tempfile.mkdtemp(prefix=f"neuralopt-docs-{tag}-"))
    shutil.rmtree(worktree)
    added = _run(["git", "worktree", "add", "--detach", str(worktree), tag], repo)
    if added.returncode != 0:
        return {**record, "deployed": False, "error": added.stderr.strip()}
    try:
        config = worktree / OVERLAY_NAME
        config.write_text(overlay_config((worktree / "zensical.toml").read_text()))
        strict = _run(["zensical", "build", "--strict", "-f", OVERLAY_NAME], worktree)
        record["strict"] = strict.returncode == 0
        if strict.returncode != 0:
            record["strict_error"] = (strict.stderr or strict.stdout).strip().splitlines()[-1:]
        deploy = ["mike", "deploy", "-F", OVERLAY_NAME, "-b", branch,
                  "--title", tag.removeprefix("v"), minor(tag)]
        if push:
            deploy.insert(2, "--push")
        done = _run(deploy, worktree)
        record["deployed"] = done.returncode == 0
        if done.returncode != 0:
            record["error"] = (done.stderr or done.stdout).strip().splitlines()[-3:]
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)], repo)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tags", nargs="*", default=DEFAULT_TAGS)
    parser.add_argument("--push", action="store_true", help="push gh-pages to origin")
    parser.add_argument("--branch", default="gh-pages")
    parser.add_argument("--report", type=Path, help="also write the JSON report here")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    for tool in ("zensical", "mike"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} not on PATH; run with `uv run --group docs`")
    records = [backfill(tag, repo, args.push, args.branch) for tag in args.tags]
    report = json.dumps(records, indent=2)
    print(report)
    if args.report:
        args.report.write_text(report + "\n")
    return 0 if all(r.get("deployed") for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
