from __future__ import annotations

from pathlib import Path


def test_resolve_wiki_root_uses_default_user_namespace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENKIMO_WIKI_ROOT", raising=False)
    monkeypatch.setenv("OPENKIMO_APP_DATA_DIR", str(tmp_path))

    from kimi_cli.wiki.paths import resolve_wiki_root

    assert resolve_wiki_root() == tmp_path / "users" / "default" / "wiki"


def test_resolve_wiki_root_prefers_explicit_root(tmp_path: Path, monkeypatch) -> None:
    configured = tmp_path / "configured-wiki"
    monkeypatch.setenv("OPENKIMO_WIKI_ROOT", str(configured))

    from kimi_cli.wiki.paths import resolve_wiki_root

    assert resolve_wiki_root(app_data=tmp_path / "ignored") == configured.resolve()


def test_wiki_schema_version_is_one() -> None:
    from kimi_cli.wiki.paths import WIKI_SCHEMA_VERSION

    assert WIKI_SCHEMA_VERSION == 1
