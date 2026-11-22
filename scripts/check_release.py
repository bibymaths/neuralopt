"""Release metadata consistency check.

Usage::

    python scripts/check_release.py                 # every PR: metadata of the current version
    python scripts/check_release.py v0.7.0          # a release tag: also tag == version
    python scripts/check_release.py v0.3.0 --backfill   # retroactive tag: notes + changelog only
    python scripts/check_release.py v0.7.0 --pypi-allowed   # prints true/false

``pyproject.toml`` is the single source of the version. ``CITATION.cff`` must carry the
same version and a ``date-released`` equal to that version's CHANGELOG date;
``.zenodo.json`` must be valid and either omit ``version`` or match it. Every checked
version needs ``docs/release-notes/vX.Y.Z.md`` and a ``## [X.Y.Z] - YYYY-MM-DD`` CHANGELOG
heading. ``CITATION.cff`` schema validation is done separately with ``cffconvert``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

from packaging.version import Version

FIRST_PYPI_VERSION = Version("0.7.0")
ZENODO_REQUIRED = ("title", "upload_type", "description", "creators", "license",
                   "access_right", "related_identifiers")


def _cff_field(text: str, field: str) -> str | None:
    match = re.search(rf"^{field}:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", text, re.MULTILINE)
    return match.group(1) if match else None


def _changelog_date(changelog: str, version: str) -> str | None:
    match = re.search(rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})\s*$",
                      changelog, re.MULTILINE)
    return match.group(1) if match else None


def _version_documented(root: Path, version: str) -> list[str]:
    errors = []
    if not (root / "docs" / "release-notes" / f"v{version}.md").is_file():
        errors.append(f"missing release notes docs/release-notes/v{version}.md")
    if _changelog_date((root / "CHANGELOG.md").read_text(), version) is None:
        errors.append(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' heading")
    return errors


def _zenodo_errors(root: Path, version: str) -> list[str]:
    try:
        zenodo = json.loads((root / ".zenodo.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f".zenodo.json is not valid JSON: {exc}"]
    errors = [f".zenodo.json is missing '{key}'" for key in ZENODO_REQUIRED if key not in zenodo]
    if zenodo.get("upload_type", "software") != "software":
        errors.append(".zenodo.json upload_type must be 'software'")
    if "version" in zenodo and zenodo["version"] != version:
        errors.append(f".zenodo.json version {zenodo['version']} != pyproject {version}")
    creators = zenodo.get("creators")
    if not isinstance(creators, list) or not creators or not all(
        isinstance(c, dict) and c.get("name") for c in creators
    ):
        errors.append(".zenodo.json creators must be a non-empty list of named entries")
    return errors


def check(root: Path, tag: str | None = None, backfill: bool = False) -> list[str]:
    """Return a list of problems (empty when consistent)."""
    root = Path(root)
    if backfill:
        if tag is None:
            return ["--backfill needs a tag"]
        return _version_documented(root, tag.removeprefix("v"))
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    errors = []
    if tag is not None and tag.removeprefix("v") != version:
        errors.append(f"tag {tag} != pyproject version {version}")
    cff = (root / "CITATION.cff").read_text()
    cff_version = _cff_field(cff, "version")
    if cff_version != version:
        errors.append(f"CITATION.cff version {cff_version} != pyproject {version}")
    errors += _version_documented(root, version)
    released = _cff_field(cff, "date-released")
    expected = _changelog_date((root / "CHANGELOG.md").read_text(), version)
    if expected is not None and released != expected:
        errors.append(f"CITATION.cff date-released {released} != CHANGELOG date {expected}")
    errors += _zenodo_errors(root, version)
    return errors


def pypi_allowed(tag: str) -> bool:
    return Version(tag.removeprefix("v")) >= FIRST_PYPI_VERSION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", nargs="?")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--pypi-allowed", action="store_true",
                        help="print whether this tag may be published to PyPI and exit")
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args(argv)
    if args.pypi_allowed:
        if args.tag is None:
            parser.error("--pypi-allowed needs a tag")
        print("true" if pypi_allowed(args.tag) else "false")
        return 0
    errors = check(args.root, args.tag, args.backfill)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not errors:
        print("release metadata consistent")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
