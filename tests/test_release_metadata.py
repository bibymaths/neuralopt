"""Release metadata: one version source and a consistency check run on every PR."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tomllib
from pathlib import Path

import pytest

import neuralopt

ROOT = Path(__file__).parents[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_release", ROOT / "scripts" /
                                                  "check_release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_release = _load_checker()
PYPROJECT_VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def _copy_metadata(tmp_path: Path) -> Path:
    for name in ("pyproject.toml", "CITATION.cff", ".zenodo.json", "CHANGELOG.md"):
        shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "docs" / "release-notes", tmp_path / "docs" / "release-notes")
    return tmp_path


def test_version_comes_from_package_metadata():
    assert neuralopt.__version__ == PYPROJECT_VERSION


def test_repository_metadata_is_consistent():
    assert check_release.check(ROOT) == []
    assert check_release.check(ROOT, tag=f"v{PYPROJECT_VERSION}") == []


def test_every_released_version_has_notes_and_a_changelog_entry():
    for minor in range(1, 8):
        assert check_release.check(ROOT, tag=f"v0.{minor}.0", backfill=True) == []


def test_tag_must_match_pyproject(tmp_path):
    root = _copy_metadata(tmp_path)
    errors = check_release.check(root, tag="v9.9.9")
    assert any("pyproject" in e for e in errors)


def test_citation_version_must_match(tmp_path):
    root = _copy_metadata(tmp_path)
    cff = root / "CITATION.cff"
    cff.write_text(cff.read_text().replace(f"version: {PYPROJECT_VERSION}", "version: 0.0.1"))
    assert any("CITATION.cff version" in e for e in check_release.check(root))


def test_citation_date_must_match_changelog(tmp_path):
    root = _copy_metadata(tmp_path)
    cff = root / "CITATION.cff"
    text = "\n".join(
        "date-released: 1999-01-01" if line.startswith("date-released:") else line
        for line in cff.read_text().splitlines()
    )
    cff.write_text(text + "\n")
    assert any("date-released" in e for e in check_release.check(root))


def test_missing_release_notes_fail(tmp_path):
    root = _copy_metadata(tmp_path)
    (root / "docs" / "release-notes" / f"v{PYPROJECT_VERSION}.md").unlink()
    assert any("release notes" in e for e in check_release.check(root))


def test_missing_changelog_entry_fails(tmp_path):
    root = _copy_metadata(tmp_path)
    log = root / "CHANGELOG.md"
    log.write_text(log.read_text().replace(f"## [{PYPROJECT_VERSION}]", "## [x.y.z]"))
    assert any("CHANGELOG" in e for e in check_release.check(root))


@pytest.mark.parametrize("mutate", [
    lambda z: z.pop("creators"),
    lambda z: z.update(upload_type="dataset"),
    lambda z: z.update(version="0.0.1"),
])
def test_invalid_zenodo_metadata_fails(tmp_path, mutate):
    root = _copy_metadata(tmp_path)
    zenodo = json.loads((root / ".zenodo.json").read_text())
    mutate(zenodo)
    (root / ".zenodo.json").write_text(json.dumps(zenodo))
    assert any(".zenodo.json" in e for e in check_release.check(root))


def test_malformed_zenodo_json_fails(tmp_path):
    root = _copy_metadata(tmp_path)
    (root / ".zenodo.json").write_text("{not json")
    assert any(".zenodo.json" in e for e in check_release.check(root))


def test_backfill_ignores_the_current_version_but_needs_notes(tmp_path):
    root = _copy_metadata(tmp_path)
    assert check_release.check(root, tag="v0.3.0", backfill=True) == []
    (root / "docs" / "release-notes" / "v0.3.0.md").unlink()
    assert check_release.check(root, tag="v0.3.0", backfill=True)


def test_pypi_is_allowed_only_from_v0_7_0():
    assert not check_release.pypi_allowed("v0.6.0")
    assert check_release.pypi_allowed("v0.7.0")
    assert check_release.pypi_allowed("v1.0.0")


def test_title_and_author_agree_across_metadata():
    zenodo = json.loads((ROOT / ".zenodo.json").read_text())
    cff = (ROOT / "CITATION.cff").read_text()
    title = zenodo["title"]
    assert f'title: "{title}"' in cff
    description = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["description"]
    assert title.removeprefix("NeuralOpt: ").lower() == description.rstrip(".").lower()
    assert zenodo["creators"][0]["orcid"] == "0009-0005-3179-7408"
    assert "0009-0005-3179-7408" in cff


def test_citing_page_agrees_with_citation_cff():
    page = (ROOT / "docs" / "citing.md").read_text()
    assert f"version = {{{PYPROJECT_VERSION}}}" in page
    assert "0009-0005-3179-7408" in page
    zenodo = json.loads((ROOT / ".zenodo.json").read_text())
    assert zenodo["title"] in page


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_body_rewrites_relative_links_to_the_docs_site():
    body = _load_script("release_body").absolutize(
        "See [report](../research/gram-space-v0.7.md#result), "
        "[previous](v0.6.0.md), [index](index.md), [log](../changelog.md), "
        "[ext](https://example.org/a.md) and [code](../../src/neuralopt/compiled.py).",
        base="https://bibymaths.github.io/neuralopt/latest/",
    )
    site = "https://bibymaths.github.io/neuralopt/latest/"
    assert f"[report]({site}research/gram-space-v0.7/#result)" in body
    assert f"[previous]({site}release-notes/v0.6.0/)" in body
    assert f"[index]({site}release-notes/)" in body
    assert f"[log]({site}changelog/)" in body
    assert "[ext](https://example.org/a.md)" in body
    blob = "https://github.com/bibymaths/neuralopt/blob/main/"
    assert f"[code]({blob}src/neuralopt/compiled.py)" in body


def test_backfill_overlay_adds_versioning_without_touching_the_rest():
    overlay = _load_script("backfill_docs").overlay_config(
        '[project]\nsite_name = "neuralopt"\ndocs_dir = "docs"\n\n'
        '[project.theme]\nname = "material"\n'
    )
    parsed = tomllib.loads(overlay)
    assert parsed["project"]["site_name"] == "neuralopt"
    assert parsed["project"]["site_url"] == "https://bibymaths.github.io/neuralopt/"
    assert parsed["project"]["theme"] == {"name": "material"}
    assert parsed["project"]["extra"]["version"] == {"provider": "mike", "default": "latest",
                                                    "alias": True}


def test_backfill_overlay_refuses_configs_it_would_corrupt():
    module = _load_script("backfill_docs")
    with pytest.raises(ValueError):
        module.overlay_config('[project]\nsite_url = "x"\n')
    with pytest.raises(ValueError):
        module.overlay_config('[project]\n[project.extra.version]\nprovider = "mike"\n')


def test_backfill_minor_versions():
    module = _load_script("backfill_docs")
    assert module.minor("v0.3.0") == "0.3"
    assert module.DEFAULT_TAGS == [f"v0.{i}.0" for i in range(1, 7)]
