"""Stable workspace registration for portable Wiki source provenance."""

from __future__ import annotations

import contextlib
import json
import os
import stat
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from uuid import UUID, uuid4

from kimi_cli.utils.io import atomic_json_write
from kimi_cli.wiki.models import SourceRef, validate_relative_source_path
from kimi_cli.wiki.schema import content_hash

WORKSPACE_IDENTITY_MARKER = ".openkimo-workspace.json"
WORKSPACE_IDENTITY_SCHEMA_VERSION = 1
WORKSPACE_REGISTRY_SCHEMA_VERSION = 1


class WorkspaceRecord(TypedDict):
    """One registry-only absolute workspace mapping."""

    path: str
    last_seen_at: str


class RegistryData(TypedDict):
    """Versioned registry data, separate from portable Wiki provenance."""

    schema_version: Literal[1]
    workspaces: dict[str, WorkspaceRecord]


class UnsupportedWorkspaceRegistrySchema(ValueError):
    """Raised when a registry was written by unsupported future code."""


class WorkspaceRegistry:
    """Persist canonical workspace roots separately from portable Wiki pages."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()

    def register(self, path: Path, *, workspace_id: UUID | None = None) -> UUID:
        """Register a workspace root, retaining a supplied UUID after a move."""
        canonical = _canonical_directory(path)
        with self._locked():
            data = self._read()
            existing = self._id_for_path(data, canonical)
            marker_id = _read_workspace_identity(canonical)
            requested = marker_id or workspace_id
            if existing is not None and requested is not None and existing != requested:
                raise ValueError("workspace identity marker does not match the registry")
            key = existing or requested or uuid4()
            previous = data["workspaces"].get(str(key))
            if previous is not None and Path(previous["path"]).resolve(strict=False) != canonical:
                previous_root = Path(previous["path"])
                if previous_root.is_dir():
                    raise ValueError(
                        "workspace identity is already registered at another directory"
                    )
            _ensure_workspace_identity(canonical, key)
            data["workspaces"][str(key)] = {
                "path": str(canonical),
                "last_seen_at": datetime.now().astimezone().isoformat(),
            }
            atomic_json_write(data, self.path)
        return key

    def relative_source(self, workspace_id: UUID, path: Path) -> SourceRef:
        """Return source provenance using only a UUID and POSIX-relative path."""
        root = self._registered_root(workspace_id)
        if root is None:
            raise ValueError("workspace is not registered or is unavailable")
        try:
            candidate = path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError("workspace source file must exist") from exc
        if not candidate.is_file() or not candidate.is_relative_to(root):
            raise ValueError("workspace source file must be inside its registered workspace")
        return SourceRef(
            kind="workspace-file",
            workspace_id=workspace_id,
            path=candidate.relative_to(root).as_posix(),
            content_hash=content_hash(candidate.read_bytes()),
        )

    def resolve(self, source: SourceRef) -> Path | None:
        """Resolve a portable workspace source only when it remains contained and present."""
        if source.kind != "workspace-file" or source.workspace_id is None or source.path is None:
            return None
        root = self._registered_root(source.workspace_id)
        if root is None or not _is_safe_relative_source_path(source.path):
            return None
        try:
            candidate = (root / source.path).resolve(strict=True)
        except OSError:
            return None
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
        return candidate

    def _registered_root(self, workspace_id: UUID) -> Path | None:
        with self._locked():
            data = self._read()
            record = data["workspaces"].get(str(workspace_id))
        if record is None:
            return None
        try:
            root = Path(record["path"]).resolve(strict=True)
        except OSError:
            return None
        return root if root.is_dir() else None

    def _read(self) -> RegistryData:
        if not self.path.exists():
            return _empty_registry()
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("workspace registry must be a regular file")
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("workspace registry must be valid UTF-8 JSON") from exc
        data, migrated = _validate_registry(parsed)
        if migrated:
            atomic_json_write(data, self.path)
        return data

    @staticmethod
    def _id_for_path(data: RegistryData, path: Path) -> UUID | None:
        for raw_id, record in data["workspaces"].items():
            if Path(record["path"]).resolve(strict=False) == path:
                return UUID(raw_id)
        return None

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialize registry replacement without sharing the later Wiki writer lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _canonical_directory(path: Path) -> Path:
    try:
        canonical = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("workspace directory must exist") from exc
    if not canonical.is_dir():
        raise ValueError("workspace must be a directory")
    return canonical


def _identity_marker_path(root: Path) -> Path:
    return root / WORKSPACE_IDENTITY_MARKER


def _read_workspace_identity(root: Path) -> UUID | None:
    marker = _identity_marker_path(root)
    try:
        marker_stat = marker.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(marker_stat.st_mode):
        raise ValueError("workspace identity marker must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(marker, flags)
    except OSError as exc:
        raise ValueError("workspace identity marker cannot be opened safely") from exc
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as input_file:
            parsed = json.load(input_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("workspace identity marker must be valid UTF-8 JSON") from exc
    return _validate_workspace_identity(parsed)


def _ensure_workspace_identity(root: Path, workspace_id: UUID) -> None:
    existing = _read_workspace_identity(root)
    if existing is not None:
        if existing != workspace_id:
            raise ValueError("workspace identity marker does not match the registry")
        return
    marker = _identity_marker_path(root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(marker, flags, 0o600)
    except FileExistsError as exc:
        existing = _read_workspace_identity(root)
        if existing != workspace_id:
            raise ValueError("workspace identity marker does not match the registry") from exc
        return
    except OSError as exc:
        raise ValueError("workspace identity marker cannot be created safely") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        json.dump(
            {
                "schema_version": WORKSPACE_IDENTITY_SCHEMA_VERSION,
                "workspace_id": str(workspace_id),
            },
            output,
            separators=(",", ":"),
        )
        output.flush()
        os.fsync(output.fileno())


def _validate_workspace_identity(parsed: object) -> UUID:
    if not isinstance(parsed, dict):
        raise ValueError("workspace identity marker has an invalid shape")
    data = cast(dict[str, object], parsed)
    if set(data) != {"schema_version", "workspace_id"}:
        raise ValueError("workspace identity marker has an invalid shape")
    schema_version = data["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != WORKSPACE_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError("workspace identity marker schema version is unsupported")
    workspace_id = data["workspace_id"]
    if not isinstance(workspace_id, str):
        raise ValueError("workspace identity marker has an invalid workspace UUID")
    try:
        return UUID(workspace_id)
    except ValueError as exc:
        raise ValueError("workspace identity marker has an invalid workspace UUID") from exc


def _is_safe_relative_source_path(value: str) -> bool:
    try:
        validate_relative_source_path(value)
    except ValueError:
        return False
    return True


def _empty_registry() -> RegistryData:
    return {"schema_version": WORKSPACE_REGISTRY_SCHEMA_VERSION, "workspaces": {}}


def _validate_registry(parsed: object) -> tuple[RegistryData, bool]:
    if not isinstance(parsed, dict):
        raise ValueError("workspace registry has an invalid shape")
    container = cast(dict[str, object], parsed)
    keys = set(container)
    if keys == {"workspaces"}:
        schema_version = WORKSPACE_REGISTRY_SCHEMA_VERSION
        migrated = True
    elif keys == {"schema_version", "workspaces"}:
        schema_version = container["schema_version"]
        migrated = False
    else:
        raise ValueError("workspace registry has an invalid shape")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("workspace registry schema version must be an integer")
    if schema_version != WORKSPACE_REGISTRY_SCHEMA_VERSION:
        raise UnsupportedWorkspaceRegistrySchema(
            f"workspace registry schema version is unsupported: {schema_version}"
        )
    workspaces_value = container["workspaces"]
    if not isinstance(workspaces_value, dict):
        raise ValueError("workspace registry has an invalid shape")
    workspaces = cast(dict[object, object], workspaces_value)
    validated: dict[str, WorkspaceRecord] = {}
    for raw_id, raw_record in workspaces.items():
        if not isinstance(raw_id, str) or not isinstance(raw_record, dict):
            raise ValueError("workspace registry has an invalid shape")
        try:
            UUID(raw_id)
        except ValueError as exc:
            raise ValueError("workspace registry contains an invalid UUID") from exc
        record = cast(dict[str, Any], raw_record)
        if set(record) != {"path", "last_seen_at"}:
            raise ValueError("workspace registry has an invalid shape")
        path, last_seen_at = record["path"], record["last_seen_at"]
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not isinstance(last_seen_at, str)
        ):
            raise ValueError("workspace registry has an invalid shape")
        try:
            timestamp = datetime.fromisoformat(last_seen_at)
        except ValueError as exc:
            raise ValueError("workspace registry has an invalid timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError("workspace registry timestamps must include a timezone")
        validated[raw_id] = {"path": path, "last_seen_at": last_seen_at}
    return (
        {"schema_version": WORKSPACE_REGISTRY_SCHEMA_VERSION, "workspaces": validated},
        migrated,
    )
