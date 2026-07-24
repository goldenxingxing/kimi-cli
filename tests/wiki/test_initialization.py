from __future__ import annotations

import importlib.resources
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

import kimi_cli.wiki.initialize as wiki_initialize
from kimi_cli.wiki.initialize import UnsupportedWikiSchema, ensure_wiki
from kimi_cli.wiki.paths import WIKI_SCHEMA_VERSION


def test_packaged_templates_are_generic() -> None:
    templates = importlib.resources.files("kimi_cli.wiki").joinpath("templates")
    text = "\n".join(
        templates.joinpath(name).read_text(encoding="utf-8")
        for name in ("schema.md", "index.md", "overview.md", "log.md", "manifest.json")
    )

    assert "comparisons" in text
    for forbidden in ("local_agent_work", "/Users/qunwei", "Obsidian", "TaskOutput"):
        assert forbidden not in text


def test_packaged_manifest_declares_default_namespace() -> None:
    manifest = importlib.resources.files("kimi_cli.wiki").joinpath("templates", "manifest.json")

    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "namespace": "default",
    }


def test_wheel_includes_wiki_templates(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=package_root,
        check=True,
    )

    wheel = next(tmp_path.glob("kimi_cli-*.whl"))
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())

    assert {
        "kimi_cli/wiki/templates/schema.md",
        "kimi_cli/wiki/templates/index.md",
        "kimi_cli/wiki/templates/overview.md",
        "kimi_cli/wiki/templates/log.md",
        "kimi_cli/wiki/templates/manifest.json",
    } <= names


def test_ensure_wiki_is_idempotent_and_preserves_markdown(tmp_path: Path) -> None:
    layout = ensure_wiki(tmp_path / "wiki")
    layout.index.write_text("# My edited index\n", encoding="utf-8")

    second = ensure_wiki(layout.root)

    assert second == layout
    assert second.index.read_text(encoding="utf-8") == "# My edited index\n"
    assert {path.name for path in layout.root.iterdir()} >= {
        "schema.md",
        "index.md",
        "overview.md",
        "log.md",
        "entities",
        "concepts",
        "comparisons",
        "sources",
        "queries",
        "lint",
        ".openkimo",
    }
    assert {path.name for path in layout.metadata.iterdir()} >= {
        "manifest.json",
        "revision",
        "journal",
        "locks",
    }
    assert layout.revision.read_text(encoding="ascii") == "0\n"
    assert layout.database == layout.metadata / "search.sqlite3"


def test_ensure_wiki_rejects_future_manifest_schema_without_changing_markdown(
    tmp_path: Path,
) -> None:
    layout = ensure_wiki(tmp_path / "wiki")
    layout.overview.write_text("# User-owned overview\n", encoding="utf-8")
    layout.metadata.joinpath("manifest.json").write_text(
        json.dumps({"schema_version": WIKI_SCHEMA_VERSION + 1, "namespace": "default"}),
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedWikiSchema):
        ensure_wiki(layout.root)

    assert layout.overview.read_text(encoding="utf-8") == "# User-owned overview\n"


def test_ensure_wiki_runs_only_explicit_metadata_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = ensure_wiki(tmp_path / "wiki")
    layout.log.write_text("# User-owned audit\n", encoding="utf-8")

    def migrate_v1_to_v2(migration_layout, manifest):
        assert migration_layout == layout
        assert manifest == {"schema_version": 1, "namespace": "default"}
        return {"schema_version": 2, "namespace": "default"}

    monkeypatch.setattr(wiki_initialize, "WIKI_SCHEMA_VERSION", 2)
    monkeypatch.setattr(wiki_initialize, "_METADATA_MIGRATIONS", {1: migrate_v1_to_v2})

    ensure_wiki(layout.root)

    assert json.loads(layout.metadata.joinpath("manifest.json").read_text(encoding="utf-8")) == {
        "schema_version": 2,
        "namespace": "default",
    }
    assert layout.log.read_text(encoding="utf-8") == "# User-owned audit\n"
