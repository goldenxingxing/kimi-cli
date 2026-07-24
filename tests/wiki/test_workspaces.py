from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from kimi_cli.wiki.models import SourceRef
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.workspaces import WorkspaceRegistry


@pytest.fixture
def registry(tmp_path: Path) -> WorkspaceRegistry:
    return WorkspaceRegistry(tmp_path / ".openkimo" / "workspaces.json")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def test_register_records_one_canonical_workspace_path(
    registry: WorkspaceRegistry, workspace: Path
) -> None:
    workspace_id = registry.register(workspace)

    record = json.loads(registry.path.read_text(encoding="utf-8"))["workspaces"][str(workspace_id)]

    assert UUID(str(workspace_id)) == workspace_id
    assert record["path"] == str(workspace.resolve())
    assert datetime.fromisoformat(record["last_seen_at"]).tzinfo is not None
    assert registry.register(workspace) == workspace_id


def test_registry_writes_current_schema_version(
    registry: WorkspaceRegistry, workspace: Path
) -> None:
    workspace_id = registry.register(workspace)

    data = json.loads(registry.path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["workspaces"][str(workspace_id)]["path"] == str(workspace.resolve())
    assert json.loads((workspace / ".openkimo-workspace.json").read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "workspace_id": str(workspace_id),
    }


def test_registry_migrates_explicit_legacy_unversioned_shape(
    registry: WorkspaceRegistry, workspace: Path
) -> None:
    workspace_id = uuid4()
    registry.path.parent.mkdir()
    registry.path.write_text(
        json.dumps(
            {
                "workspaces": {
                    str(workspace_id): {
                        "path": str(workspace.resolve()),
                        "last_seen_at": "2026-07-24T12:00:00+00:00",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert registry.register(workspace) == workspace_id
    assert json.loads(registry.path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_registry_rejects_future_schema_without_rewriting_it(
    registry: WorkspaceRegistry, workspace: Path
) -> None:
    registry.path.parent.mkdir()
    future = json.dumps({"schema_version": 2, "workspaces": {}}, separators=(",", ":"))
    registry.path.write_text(future, encoding="utf-8")

    with pytest.raises(ValueError, match="schema version is unsupported"):
        registry.register(workspace)

    assert registry.path.read_text(encoding="utf-8") == future


def test_workspace_move_updates_only_registry(registry: WorkspaceRegistry, tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    workspace_id = registry.register(old)
    page = tmp_path / "authoritative-page.md"
    page.write_text("unchanged", encoding="utf-8")
    moved = tmp_path / "moved"
    old.rename(moved)
    fresh_registry = WorkspaceRegistry(registry.path)

    assert fresh_registry.register(moved) == workspace_id
    assert json.loads(registry.path.read_text(encoding="utf-8"))["workspaces"][str(workspace_id)][
        "path"
    ] == str(moved.resolve())
    assert page.read_text(encoding="utf-8") == "unchanged"


def test_register_rejects_symlinked_workspace_identity_marker(
    registry: WorkspaceRegistry, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "identity.json"
    outside.write_text('{"schema_version":1,"workspace_id":"00000000-0000-0000-0000-000000000000"}')
    (workspace / ".openkimo-workspace.json").symlink_to(outside)

    with pytest.raises(ValueError, match="identity marker"):
        registry.register(workspace)


@pytest.mark.parametrize(
    "marker",
    [
        {"schema_version": 2, "workspace_id": "00000000-0000-0000-0000-000000000000"},
        {"schema_version": True, "workspace_id": "00000000-0000-0000-0000-000000000000"},
        {"schema_version": 1.0, "workspace_id": "00000000-0000-0000-0000-000000000000"},
        {"schema_version": 1, "workspace_id": "not-a-uuid"},
        {"schema_version": 1, "workspace_id": "00000000-0000-0000-0000-000000000000", "x": 1},
    ],
)
def test_register_rejects_tampered_or_future_workspace_identity_marker(
    registry: WorkspaceRegistry, workspace: Path, marker: dict[str, object]
) -> None:
    marker_path = workspace / ".openkimo-workspace.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="identity marker"):
        registry.register(workspace)


def test_marker_cannot_hijack_live_workspace_identity(
    registry: WorkspaceRegistry, workspace: Path, tmp_path: Path
) -> None:
    workspace_id = registry.register(workspace)
    other = tmp_path / "other-workspace"
    other.mkdir()
    (other / ".openkimo-workspace.json").write_text(
        json.dumps({"schema_version": 1, "workspace_id": str(workspace_id)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.register(other)

    assert registry.register(workspace) == workspace_id


def test_relative_source_is_portable_and_resolves_registered_file(
    registry: WorkspaceRegistry, workspace: Path
) -> None:
    source_file = workspace / "docs" / "source.txt"
    source_file.parent.mkdir()
    source_file.write_text("global wiki", encoding="utf-8")
    workspace_id = registry.register(workspace)

    source = registry.relative_source(workspace_id, source_file)

    assert source.kind == "workspace-file"
    assert source.workspace_id == workspace_id
    assert source.path == "docs/source.txt"
    assert source.content_hash == content_hash(b"global wiki")
    assert registry.resolve(source) == source_file.resolve()


@pytest.mark.parametrize("source_path", ["../secret", "/tmp/secret", r"C:\\secret", "missing.txt"])
def test_unknown_missing_and_escape_sources_are_not_executable(
    registry: WorkspaceRegistry, workspace: Path, source_path: str
) -> None:
    workspace_id = registry.register(workspace)
    source = SourceRef.model_construct(
        kind="workspace-file",
        workspace_id=workspace_id,
        path=source_path,
        session_id=None,
        url=None,
        content_hash="sha256:" + "a" * 64,
    )

    assert registry.resolve(source) is None


def test_unregistered_workspace_and_escape_file_are_rejected(
    registry: WorkspaceRegistry, workspace: Path, tmp_path: Path
) -> None:
    source_file = workspace / "inside.txt"
    source_file.write_text("inside", encoding="utf-8")
    unknown_id = uuid4()

    with pytest.raises(ValueError, match="not registered"):
        registry.relative_source(unknown_id, source_file)

    workspace_id = registry.register(workspace)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="must be inside"):
        registry.relative_source(workspace_id, outside)


def test_symlinked_workspace_source_cannot_escape_registry_root(
    registry: WorkspaceRegistry, workspace: Path, tmp_path: Path
) -> None:
    workspace_id = registry.register(workspace)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    escaped = workspace / "escaped.txt"
    escaped.symlink_to(outside)
    source = SourceRef(
        kind="workspace-file",
        workspace_id=workspace_id,
        path="escaped.txt",
        content_hash=content_hash(b"outside"),
    )

    assert registry.resolve(source) is None
    with pytest.raises(ValueError, match="must be inside"):
        registry.relative_source(workspace_id, escaped)


def test_registry_rejects_non_directory_workspace(
    registry: WorkspaceRegistry, tmp_path: Path
) -> None:
    file = tmp_path / "not-a-workspace.txt"
    file.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="directory"):
        registry.register(file)
