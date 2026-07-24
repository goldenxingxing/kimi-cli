"""Idempotently create and validate the user-owned global Wiki layout."""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from kimi_cli.wiki.paths import WIKI_SCHEMA_VERSION, resolve_wiki_root

CATEGORY_DIRS = ("entities", "concepts", "comparisons", "sources", "queries", "lint")
SPECIAL_FILES = ("schema.md", "index.md", "overview.md", "log.md")
_MANIFEST_NAME = "manifest.json"
_NAMESPACE = "default"


class UnsupportedWikiSchema(ValueError):
    """Raised when installed code cannot safely open a Wiki schema version."""

    def __init__(self, schema_version: int) -> None:
        self.schema_version = schema_version
        super().__init__(f"unsupported Wiki schema version: {schema_version}")


@dataclass(frozen=True, slots=True)
class WikiLayout:
    """Canonical paths for the shared, user-owned Wiki and its private metadata."""

    root: Path
    index: Path
    overview: Path
    log: Path
    metadata: Path
    revision: Path
    database: Path


MetadataMigration = Callable[[WikiLayout, Mapping[str, Any]], Mapping[str, Any]]
# Future metadata upgrades must add an explicit migration from N to N + 1 here.
_METADATA_MIGRATIONS: dict[int, MetadataMigration] = {}


def layout_for(root: Path) -> WikiLayout:
    """Return canonical managed paths without creating or mutating anything."""
    canonical_root = root.expanduser().resolve(strict=False)
    metadata = canonical_root / ".openkimo"
    return WikiLayout(
        root=canonical_root,
        index=canonical_root / "index.md",
        overview=canonical_root / "overview.md",
        log=canonical_root / "log.md",
        metadata=metadata,
        revision=metadata / "revision",
        database=metadata / "search.sqlite3",
    )


def ensure_wiki(root: Path | None = None) -> WikiLayout:
    """Create missing Wiki files only, while preserving all existing Markdown."""
    layout = layout_for(root or resolve_wiki_root())
    _ensure_directory(layout.root, layout.root)
    _ensure_directory(layout.metadata, layout.root)
    manifest_path = layout.metadata / _MANIFEST_NAME
    _copy_template_exclusive(_MANIFEST_NAME, manifest_path, layout.root)
    manifest = _read_manifest(manifest_path, layout.root)
    _migrate_metadata(layout, manifest)

    for name in CATEGORY_DIRS:
        _ensure_directory(layout.root / name, layout.root)
    for name in ("journal", "locks"):
        _ensure_directory(layout.metadata / name, layout.root)
    for name in SPECIAL_FILES:
        _copy_template_exclusive(name, layout.root / name, layout.root)
    _ensure_revision(layout.revision, layout.root)
    return layout


def _ensure_directory(path: Path, root: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Wiki managed path must be a directory: {path}")
    _assert_within_root(path, root)


def _copy_template_exclusive(name: str, destination: Path, root: Path) -> None:
    """Copy one packaged template only if the user has not created it already."""
    if destination.exists() or destination.is_symlink():
        _assert_regular_file(destination, root)
        return
    try:
        with destination.open("xb") as output:
            output.write(_template_bytes(name))
    except FileExistsError:
        _assert_regular_file(destination, root)
    else:
        _assert_regular_file(destination, root)


def _template_bytes(name: str) -> bytes:
    template = importlib.resources.files("kimi_cli.wiki").joinpath("templates", name)
    return template.read_bytes()


def _read_manifest(path: Path, root: Path) -> dict[str, Any]:
    _assert_regular_file(path, root)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Wiki manifest must be valid UTF-8 JSON") from exc
    return _validate_manifest(parsed)


def _validate_manifest(parsed: object) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Wiki manifest has an invalid shape")
    manifest = cast(dict[str, Any], parsed)
    if set(manifest) != {"schema_version", "namespace"}:
        raise ValueError("Wiki manifest has an invalid shape")
    schema_version = manifest["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise ValueError("Wiki manifest schema_version must be a positive integer")
    if manifest["namespace"] != _NAMESPACE:
        raise ValueError("Wiki manifest namespace is unsupported")
    return manifest


def _migrate_metadata(layout: WikiLayout, manifest: Mapping[str, Any]) -> None:
    schema_version = manifest["schema_version"]
    assert isinstance(schema_version, int) and not isinstance(schema_version, bool)
    while schema_version != WIKI_SCHEMA_VERSION:
        if schema_version > WIKI_SCHEMA_VERSION:
            raise UnsupportedWikiSchema(schema_version)
        migration = _METADATA_MIGRATIONS.get(schema_version)
        if migration is None:
            raise UnsupportedWikiSchema(schema_version)
        current_manifest = _validate_manifest(dict(migration(layout, manifest)))
        schema_version = current_manifest.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version < 1
        ):
            raise ValueError("Wiki metadata migration returned an invalid schema version")
        previous_version = manifest["schema_version"]
        assert isinstance(previous_version, int) and not isinstance(previous_version, bool)
        if schema_version != previous_version + 1:
            raise ValueError("Wiki metadata migrations must advance exactly one schema version")
        _write_manifest(layout.metadata / _MANIFEST_NAME, current_manifest, layout.root)
        manifest = current_manifest


def _write_manifest(path: Path, manifest: Mapping[str, Any], root: Path) -> None:
    """Atomically replace only software-owned manifest metadata during a migration."""
    _assert_regular_file(path, root)
    canonical = {
        "schema_version": manifest["schema_version"],
        "namespace": manifest["namespace"],
    }
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(json.dumps(canonical, separators=(",", ":")))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    _assert_regular_file(path, root)


def _ensure_revision(path: Path, root: Path) -> None:
    if not path.exists() and not path.is_symlink():
        try:
            with path.open("x", encoding="ascii") as output:
                output.write("0\n")
        except FileExistsError:
            pass
    _assert_regular_file(path, root)
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Wiki revision must be an ASCII integer") from exc
    if not value:
        path.write_text("0\n", encoding="ascii")
    elif not value.isascii() or not value.isdecimal():
        raise ValueError("Wiki revision must be a non-negative integer")


def _assert_regular_file(path: Path, root: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Wiki managed path must be a regular file: {path}")
    _assert_within_root(path, root)


def _assert_within_root(path: Path, root: Path) -> None:
    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"unable to resolve Wiki managed path: {path}") from exc
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Wiki managed path escapes its root: {path}")
