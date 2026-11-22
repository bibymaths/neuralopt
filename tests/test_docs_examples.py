"""The README and quickstart examples run as documented."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL)


@pytest.mark.parametrize("path", ["README.md", "docs/quickstart.md",
                                  "docs/guide/certified-mode.md"])
def test_documented_examples_run(path):
    namespace: dict = {}
    blocks = _python_blocks(ROOT / path)
    assert blocks
    for block in blocks:  # blocks build on each other, like a reader's session
        exec(compile(block, f"{path}", "exec"), namespace)


def test_readme_citation_version_matches_pyproject():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert f"version = {{{version}}}" in (ROOT / "README.md").read_text()
