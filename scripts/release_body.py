"""Print a GitHub release body from ``docs/release-notes/vX.Y.Z.md``.

Relative Markdown links in the notes resolve inside the docs site; on a GitHub release
page they would break. They are rewritten to absolute URLs on the published docs
(``use_directory_urls``: ``a/b.md`` -> ``a/b/``); links that leave ``docs/`` point to the
file on GitHub.

Usage: ``python scripts/release_body.py vX.Y.Z [--base URL] > body.md``
"""

from __future__ import annotations

import argparse
import posixpath
import re
from pathlib import Path

DOCS_SITE = "https://bibymaths.github.io/neuralopt/latest/"
REPO_BLOB = "https://github.com/bibymaths/neuralopt/blob/main/"
NOTES_DIR = "release-notes"  # location of the notes inside docs/
_LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)\s]+)\)")


def _rewrite(target: str, base: str) -> str:
    path, _, anchor = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(NOTES_DIR, path))
    if resolved.startswith("../"):  # outside docs/: link to the repository file
        url = REPO_BLOB + resolved.removeprefix("../")
    elif resolved.endswith(".md"):
        page = resolved.removesuffix(".md")
        page = page.removesuffix("index") if page.endswith("index") else page + "/"
        url = base + page
    else:
        url = base + resolved
    return url + (f"#{anchor}" if anchor else "")


def absolutize(markdown: str, base: str = DOCS_SITE) -> str:
    return _LINK.sub(lambda m: f"]({_rewrite(m.group(1), base)})", markdown)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag")
    parser.add_argument("--base", default=DOCS_SITE)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args(argv)
    notes = args.root / "docs" / NOTES_DIR / f"{args.tag}.md"
    print(absolutize(notes.read_text(), args.base), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
