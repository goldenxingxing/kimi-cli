from __future__ import annotations

import importlib.resources
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile


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
