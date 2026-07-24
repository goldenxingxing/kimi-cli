"""Durable, recoverable transaction boundary for authoritative Wiki Markdown."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from kimi_cli.wiki.initialize import WikiLayout
from kimi_cli.wiki.locking import WikiLock
from kimi_cli.wiki.models import PageChange
from kimi_cli.wiki.schema import content_hash, parse_page, render_page, validate_logical_page

_JOURNAL_VERSION = 1
_QUARANTINE_MARKER = "QUARANTINED"
_LOCK_TIMEOUT_SECONDS = 5.0
_HASH_PREFIX = "sha256:"


class WikiConflictError(RuntimeError):
    """Raised when the authoritative files changed after a proposal was prepared."""


class WikiRecoveryRequired(RuntimeError):
    """Raised when journal damage requires operator review before another write."""


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Summary of durable journal processing."""

    recovered_transactions: int = 0
    discarded_transactions: int = 0
    rolled_back_transactions: int = 0
    needs_reindex: bool = False
    writes_quarantined: bool = False


@dataclass(frozen=True, slots=True)
class _TargetPlan:
    kind: Literal["page", "index", "log", "revision"]
    relative_target: str
    target: Path
    old_hash: str | None
    new_hash: str
    new_bytes: bytes
    expected_page_revision: int | None = None


@dataclass(frozen=True, slots=True)
class _JournalTarget:
    kind: Literal["page", "index", "log", "revision"]
    relative_target: str
    target: Path
    old_hash: str | None
    new_hash: str
    old_artifact: Path | None
    new_artifact: Path
    expected_page_revision: int | None


@dataclass(frozen=True, slots=True)
class _Journal:
    directory: Path
    record_path: Path
    transaction_id: str
    state: Literal["prepared", "committed"]
    expected_global_revision: int
    new_revision: int
    targets: tuple[_JournalTarget, ...]


@dataclass(frozen=True, slots=True)
class WikiTransaction:
    """One fully prepared Wiki change set with optimistic revision checks."""

    layout: WikiLayout
    changes: tuple[PageChange, ...]
    expected_global_revision: int
    new_revision: int
    targets: tuple[_TargetPlan, ...]

    @classmethod
    def prepare(
        cls,
        *,
        layout: WikiLayout,
        changes: Iterable[PageChange],
        expected_global_revision: int,
        index_bytes: bytes,
        log_bytes: bytes,
    ) -> WikiTransaction:
        """Validate and capture the complete pre-approval change set in memory."""
        if isinstance(expected_global_revision, bool) or expected_global_revision < 0:
            raise ValueError("expected global revision must be a non-negative integer")
        materialized = tuple(changes)
        if not materialized:
            raise ValueError("a Wiki transaction must change at least one page")
        logical_paths = [change.page.logical_path for change in materialized]
        if len(logical_paths) != len(set(logical_paths)):
            raise ValueError("a Wiki transaction cannot change the same page twice")
        for name, data in (("index", index_bytes), ("log", log_bytes)):
            if type(data) is not bytes:
                raise TypeError(f"{name} content must be bytes")
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{name} content must be valid UTF-8") from exc

        recovery: list[RecoveryResult] = []
        lock = _wiki_lock(layout, recovery)
        with lock.shared(_LOCK_TIMEOUT_SECONDS):
            if recovery and recovery[0].writes_quarantined:
                # Reads stay available, but do not prepare a write that can never
                # safely commit without explicit operator repair.
                raise WikiRecoveryRequired("Wiki writes are quarantined pending journal repair")
            current_revision = _read_global_revision(layout)
            if current_revision != expected_global_revision:
                raise WikiConflictError(
                    f"global revision changed: expected {expected_global_revision}, "
                    f"found {current_revision}"
                )
            targets = _capture_targets(layout, materialized, index_bytes, log_bytes)

        return cls(
            layout=layout,
            changes=materialized,
            expected_global_revision=expected_global_revision,
            new_revision=expected_global_revision + 1,
            targets=targets,
        )

    def commit(self) -> int:
        """Serialize, revalidate, durably replace, and journal one Wiki mutation."""
        lock = WikiLock(_lock_path(self.layout))
        with lock.exclusive(_LOCK_TIMEOUT_SECONDS):
            recovery = _recover_transactions_locked(self.layout)
            if recovery.writes_quarantined:
                raise WikiRecoveryRequired("Wiki writes are quarantined pending journal repair")
            self._revalidate()
            journal: _Journal | None = None
            try:
                journal = self._write_prepared_journal()
            except BaseException:
                # No target can be replaced until record.json is durable. Debris
                # without that record is therefore safe to discard immediately.
                if journal is not None and not journal.record_path.exists():
                    _remove_tree_durably(journal.directory)
                raise

            for target in journal.targets:
                _durable_replace(target.target, _read_artifact(target.new_artifact))
                failpoint = f"{target.kind}_replace" if target.kind != "page" else "page_replace"
                _hit_failpoint(failpoint)
            self._mark_committed(journal)
            return self.new_revision

    @contextmanager
    def shared_read(self, timeout: float = _LOCK_TIMEOUT_SECONDS) -> Iterator[RecoveryResult]:
        """Recover under exclusive lock, then expose only a committed shared view."""
        with wiki_read_lock(self.layout, timeout=timeout) as result:
            yield result

    def _revalidate(self) -> None:
        current_revision = _read_global_revision(self.layout)
        # Validate pages first so a retried new-page transaction reports the
        # concrete collision, while still refusing all writes before journaling.
        for target in self.targets:
            current_hash = _target_hash(target.target)
            if target.kind == "page":
                if target.old_hash is None and current_hash is not None:
                    raise WikiConflictError(f"page already exists: {target.relative_target}")
                if target.old_hash is not None and current_hash != target.old_hash:
                    raise WikiConflictError(f"page revision changed: {target.relative_target}")
                if target.expected_page_revision is not None:
                    current = _read_page(target.target, target.relative_target)
                    if current.revision != target.expected_page_revision:
                        raise WikiConflictError(f"page revision changed: {target.relative_target}")
            elif current_hash != target.old_hash:
                if target.kind == "revision":
                    raise WikiConflictError(
                        f"global revision changed: expected {self.expected_global_revision}, "
                        f"found {current_revision}"
                    )
                raise WikiConflictError(f"{target.kind} changed since transaction preparation")
        if current_revision != self.expected_global_revision:
            raise WikiConflictError(
                f"global revision changed: expected {self.expected_global_revision}, "
                f"found {current_revision}"
            )

    def _write_prepared_journal(self) -> _Journal:
        journal_root = _journal_root(self.layout)
        transaction_id = uuid4().hex
        directory = journal_root / transaction_id
        directory.mkdir(mode=0o700)
        (directory / "old").mkdir(mode=0o700)
        (directory / "new").mkdir(mode=0o700)
        fsync_directory(directory)
        fsync_directory(journal_root)
        journal_targets: list[_JournalTarget] = []
        record_path = directory / "record.json"
        journal = _Journal(
            directory=directory,
            record_path=record_path,
            transaction_id=transaction_id,
            state="prepared",
            expected_global_revision=self.expected_global_revision,
            new_revision=self.new_revision,
            targets=(),
        )
        try:
            for index, target in enumerate(self.targets):
                artifact_name = f"{index:04d}"
                new_artifact = directory / "new" / artifact_name
                _durable_create(new_artifact, target.new_bytes)
                old_artifact: Path | None = None
                if target.old_hash is not None:
                    old_bytes = _read_regular(target.target)
                    if content_hash(old_bytes) != target.old_hash:
                        raise WikiConflictError(f"{target.kind} changed during journal preparation")
                    old_artifact = directory / "old" / artifact_name
                    _durable_create(old_artifact, old_bytes)
                journal_targets.append(
                    _JournalTarget(
                        kind=target.kind,
                        relative_target=target.relative_target,
                        target=target.target,
                        old_hash=target.old_hash,
                        new_hash=target.new_hash,
                        old_artifact=old_artifact,
                        new_artifact=new_artifact,
                        expected_page_revision=target.expected_page_revision,
                    )
                )
            journal = replace(journal, targets=tuple(journal_targets))
            _write_journal_record(journal)
            _hit_failpoint("journal_fsync")
            return journal
        except BaseException:
            if not record_path.exists():
                _remove_tree_durably(directory)
            raise

    @staticmethod
    def _mark_committed(journal: _Journal) -> None:
        _write_journal_record(replace(journal, state="committed"))


def recover_transactions(
    layout: WikiLayout,
    *,
    rebuild_search: Callable[[], None] | None = None,
    timeout: float = _LOCK_TIMEOUT_SECONDS,
) -> RecoveryResult:
    """Recover durable journals under the global exclusive writer lock.

    A corrupt journal sets a persistent quarantine marker and is reported rather
    than raised, keeping callers able to take a shared read-only view.
    """
    with WikiLock(_lock_path(layout)).exclusive(timeout):
        result = _recover_transactions_locked(layout)
        if result.needs_reindex and rebuild_search is not None:
            rebuild_search()
            _remove_committed_journals(layout)
        return result


@contextmanager
def wiki_read_lock(
    layout: WikiLayout,
    *,
    timeout: float = _LOCK_TIMEOUT_SECONDS,
) -> Iterator[RecoveryResult]:
    """Recover first, atomically downgrade, then hold the shared reader lock."""
    recovered: list[RecoveryResult] = []
    lock = _wiki_lock(layout, recovered)
    with lock.shared(timeout):
        yield recovered[0]


def _wiki_lock(layout: WikiLayout, recovered: list[RecoveryResult]) -> WikiLock:
    def recover() -> None:
        recovered.append(_recover_transactions_locked(layout))

    return WikiLock(_lock_path(layout), before_shared=recover)


def _capture_targets(
    layout: WikiLayout,
    changes: tuple[PageChange, ...],
    index_bytes: bytes,
    log_bytes: bytes,
) -> tuple[_TargetPlan, ...]:
    targets: list[_TargetPlan] = []
    for change in sorted(changes, key=lambda item: item.page.logical_path):
        logical_path = validate_logical_page(change.page.logical_path).as_posix()
        if change.expected_revision is None:
            if change.page.revision != 1:
                raise ValueError("new Wiki pages must start at revision one")
        elif change.page.revision != change.expected_revision + 1:
            raise ValueError("updated Wiki pages must increment revision exactly once")
        target = layout.root / logical_path
        old_hash = _target_hash(target)
        if change.expected_revision is None:
            if old_hash is not None:
                raise WikiConflictError(f"page already exists: {logical_path}")
        else:
            if old_hash is None:
                raise WikiConflictError(f"page is missing: {logical_path}")
            current = _read_page(target, logical_path)
            if current.revision != change.expected_revision:
                raise WikiConflictError(f"page revision changed: {logical_path}")
        new_bytes = render_page(change.page).encode("utf-8")
        targets.append(
            _TargetPlan(
                kind="page",
                relative_target=logical_path,
                target=target,
                old_hash=old_hash,
                new_hash=content_hash(new_bytes),
                new_bytes=new_bytes,
                expected_page_revision=change.expected_revision,
            )
        )

    for kind, relative_target, target, new_bytes in (
        ("index", "index.md", layout.index, index_bytes),
        ("log", "log.md", layout.log, log_bytes),
        ("revision", ".openkimo/revision", layout.revision, b""),
    ):
        data = (
            f"{_read_global_revision(layout) + 1}\n".encode("ascii")
            if kind == "revision"
            else new_bytes
        )
        targets.append(
            _TargetPlan(
                kind=cast(Literal["index", "log", "revision"], kind),
                relative_target=relative_target,
                target=target,
                old_hash=_target_hash(target),
                new_hash=content_hash(data),
                new_bytes=data,
            )
        )
    return tuple(targets)


def _recover_transactions_locked(layout: WikiLayout) -> RecoveryResult:
    journal_root = _journal_root(layout)
    quarantine_marker = journal_root / _QUARANTINE_MARKER
    result = RecoveryResult(writes_quarantined=quarantine_marker.exists())
    for entry in sorted(journal_root.iterdir(), key=lambda path: path.name):
        if entry.name == _QUARANTINE_MARKER:
            continue
        if entry.is_symlink() or not entry.is_dir():
            _quarantine(journal_root, entry.name)
            result = replace(result, writes_quarantined=True)
            continue
        record_path = entry / "record.json"
        if not record_path.exists():
            _remove_tree_durably(entry)
            result = replace(
                result,
                discarded_transactions=result.discarded_transactions + 1,
            )
            continue
        try:
            journal = _read_journal(layout, entry)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            _quarantine(journal_root, entry.name)
            result = replace(result, writes_quarantined=True)
            continue
        try:
            result = _recover_journal(journal, result)
        except ValueError:
            _quarantine(journal_root, entry.name)
            result = replace(result, writes_quarantined=True)
    return result


def _recover_journal(journal: _Journal, result: RecoveryResult) -> RecoveryResult:
    current_hashes = tuple(_target_hash(target.target) for target in journal.targets)
    old_hashes = tuple(target.old_hash for target in journal.targets)
    new_hashes = tuple(target.new_hash for target in journal.targets)

    if journal.state == "committed":
        if current_hashes != new_hashes:
            raise ValueError("committed Wiki journal does not match authoritative files")
        return replace(result, needs_reindex=True)

    if current_hashes == old_hashes:
        _remove_tree_durably(journal.directory)
        return replace(
            result,
            discarded_transactions=result.discarded_transactions + 1,
        )
    if current_hashes == new_hashes:
        _write_journal_record(replace(journal, state="committed"))
        return replace(
            result,
            recovered_transactions=result.recovered_transactions + 1,
            needs_reindex=True,
        )

    current_known = all(
        current in {old, new}
        for current, old, new in zip(current_hashes, old_hashes, new_hashes, strict=True)
    )
    if current_known and _artifacts_match(journal.targets, new=True):
        for target in journal.targets:
            _durable_replace(target.target, _read_artifact(target.new_artifact))
        _write_journal_record(replace(journal, state="committed"))
        return replace(
            result,
            recovered_transactions=result.recovered_transactions + 1,
            needs_reindex=True,
        )

    if not _artifacts_match(journal.targets, new=False):
        raise ValueError("Wiki transaction cannot roll forward or restore its backups")
    for target in reversed(journal.targets):
        if target.old_hash is None:
            _durable_remove(target.target)
        else:
            assert target.old_artifact is not None
            _durable_replace(target.target, _read_artifact(target.old_artifact))
    _remove_tree_durably(journal.directory)
    return replace(
        result,
        rolled_back_transactions=result.rolled_back_transactions + 1,
    )


def _read_journal(layout: WikiLayout, directory: Path) -> _Journal:
    record_path = directory / "record.json"
    raw = json.loads(_read_regular(record_path).decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Wiki journal record must be an object")
    record = cast(dict[str, Any], raw)
    if set(record) != {
        "version",
        "transaction_id",
        "state",
        "expected_global_revision",
        "new_revision",
        "targets",
    }:
        raise ValueError("Wiki journal has an invalid shape")
    transaction_id = _strict_identifier(record["transaction_id"])
    if transaction_id != directory.name:
        raise ValueError("Wiki journal transaction ID does not match its directory")
    version = _strict_non_negative_int(record["version"])
    if version != _JOURNAL_VERSION:
        raise ValueError("Wiki journal version is unsupported")
    state = record["state"]
    if state not in {"prepared", "committed"}:
        raise ValueError("Wiki journal state is invalid")
    expected_revision = _strict_non_negative_int(record["expected_global_revision"])
    new_revision = _strict_non_negative_int(record["new_revision"])
    if new_revision != expected_revision + 1:
        raise ValueError("Wiki journal revision sequence is invalid")
    raw_targets = record["targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Wiki journal targets are invalid")
    target_values = cast(list[object], raw_targets)
    targets = tuple(
        _parse_journal_target(layout, directory, index, value)
        for index, value in enumerate(target_values)
    )
    expected_order = sorted(
        target.relative_target for target in targets if target.kind == "page"
    ) + ["index.md", "log.md", ".openkimo/revision"]
    if [target.relative_target for target in targets] != expected_order:
        raise ValueError("Wiki journal target order is invalid")
    return _Journal(
        directory=directory,
        record_path=record_path,
        transaction_id=transaction_id,
        state=cast(Literal["prepared", "committed"], state),
        expected_global_revision=expected_revision,
        new_revision=new_revision,
        targets=targets,
    )


def _parse_journal_target(
    layout: WikiLayout,
    directory: Path,
    index: int,
    raw: object,
) -> _JournalTarget:
    if not isinstance(raw, dict):
        raise ValueError("Wiki journal target must be an object")
    value = cast(dict[str, Any], raw)
    if set(value) != {
        "kind",
        "target",
        "old_hash",
        "new_hash",
        "old_artifact",
        "new_artifact",
        "expected_page_revision",
    }:
        raise ValueError("Wiki journal target has an invalid shape")
    kind = value["kind"]
    if kind not in {"page", "index", "log", "revision"}:
        raise ValueError("Wiki journal target kind is invalid")
    relative_target = value["target"]
    if not isinstance(relative_target, str):
        raise ValueError("Wiki journal target path is invalid")
    target = _resolve_recorded_target(layout, cast(str, kind), relative_target)
    old_hash = _optional_hash(value["old_hash"])
    new_hash = _required_hash(value["new_hash"])
    expected_page_revision = value["expected_page_revision"]
    if expected_page_revision is not None:
        expected_page_revision = _strict_non_negative_int(expected_page_revision)
        if expected_page_revision < 1:
            raise ValueError("expected page revision must be positive")
    if kind != "page" and expected_page_revision is not None:
        raise ValueError("special Wiki targets cannot have page revisions")
    expected_artifact = f"{index:04d}"
    new_artifact = _resolve_artifact(directory, value["new_artifact"], "new", expected_artifact)
    old_artifact: Path | None = None
    if old_hash is None:
        if value["old_artifact"] is not None:
            raise ValueError("missing original cannot have a backup")
    else:
        old_artifact = _resolve_artifact(directory, value["old_artifact"], "old", expected_artifact)
    return _JournalTarget(
        kind=cast(Literal["page", "index", "log", "revision"], kind),
        relative_target=relative_target,
        target=target,
        old_hash=old_hash,
        new_hash=new_hash,
        old_artifact=old_artifact,
        new_artifact=new_artifact,
        expected_page_revision=cast(int | None, expected_page_revision),
    )


def _write_journal_record(journal: _Journal) -> None:
    targets: list[dict[str, object]] = []
    for target in journal.targets:
        targets.append(
            {
                "kind": target.kind,
                "target": target.relative_target,
                "old_hash": target.old_hash,
                "new_hash": target.new_hash,
                "old_artifact": (
                    target.old_artifact.relative_to(journal.directory).as_posix()
                    if target.old_artifact is not None
                    else None
                ),
                "new_artifact": target.new_artifact.relative_to(journal.directory).as_posix(),
                "expected_page_revision": target.expected_page_revision,
            }
        )
    record: Mapping[str, object] = {
        "version": _JOURNAL_VERSION,
        "transaction_id": journal.transaction_id,
        "state": journal.state,
        "expected_global_revision": journal.expected_global_revision,
        "new_revision": journal.new_revision,
        "targets": targets,
    }
    data = json.dumps(record, separators=(",", ":"), sort_keys=False).encode("utf-8")
    _durable_replace(journal.record_path, data)


def _resolve_recorded_target(layout: WikiLayout, kind: str, relative_target: str) -> Path:
    if kind == "page":
        return layout.root / validate_logical_page(relative_target)
    special = {
        ("index", "index.md"): layout.index,
        ("log", "log.md"): layout.log,
        ("revision", ".openkimo/revision"): layout.revision,
    }
    try:
        return special[(kind, relative_target)]
    except KeyError as exc:
        raise ValueError("Wiki journal special target is invalid") from exc


def _resolve_artifact(
    directory: Path,
    raw: object,
    expected_parent: str,
    expected_name: str,
) -> Path:
    expected = f"{expected_parent}/{expected_name}"
    if raw != expected:
        raise ValueError("Wiki journal artifact path is invalid")
    return directory / expected_parent / expected_name


def _artifacts_match(targets: tuple[_JournalTarget, ...], *, new: bool) -> bool:
    for target in targets:
        expected_hash = target.new_hash if new else target.old_hash
        artifact = target.new_artifact if new else target.old_artifact
        if expected_hash is None:
            if artifact is not None:
                return False
            continue
        if artifact is None:
            return False
        try:
            if content_hash(_read_artifact(artifact)) != expected_hash:
                return False
        except (OSError, ValueError):
            return False
    return True


def _remove_committed_journals(layout: WikiLayout) -> None:
    for entry in _journal_root(layout).iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue
        try:
            if _read_journal(layout, entry).state == "committed":
                _remove_tree_durably(entry)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            continue


def _quarantine(journal_root: Path, transaction_name: str) -> None:
    marker = journal_root / _QUARANTINE_MARKER
    if marker.exists():
        return
    data = json.dumps(
        {"reason": "unreadable-journal", "transaction": transaction_name},
        separators=(",", ":"),
    ).encode("utf-8")
    with suppress(FileExistsError):
        _durable_create(marker, data)


def _lock_path(layout: WikiLayout) -> Path:
    return layout.metadata / "locks" / "writer.lock"


def _journal_root(layout: WikiLayout) -> Path:
    root = layout.metadata / "journal"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Wiki journal root must be a real directory")
    return root


def _read_global_revision(layout: WikiLayout) -> int:
    try:
        raw = _read_regular(layout.revision).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Wiki revision must be ASCII") from exc
    if not raw or not raw.isascii() or not raw.isdecimal():
        raise ValueError("Wiki revision must be a non-negative integer")
    return int(raw)


def _read_page(path: Path, logical_path: str):
    try:
        return parse_page(_read_regular(path).decode("utf-8"), logical_path)
    except UnicodeDecodeError as exc:
        raise ValueError(f"Wiki page is not valid UTF-8: {logical_path}") from exc


def _target_hash(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    return content_hash(_read_regular(path))


def _read_artifact(path: Path) -> bytes:
    return _read_regular(path)


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if path.is_symlink():
            raise ValueError(f"Wiki transaction path must be a regular file: {path.name}") from None
        raise
    try:
        file_stat = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or (file_stat.st_dev, file_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ValueError(f"Wiki transaction path must be a regular file: {path.name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _durable_create(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def _durable_replace(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=False, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError(f"Wiki transaction parent must be a real directory: {target.parent.name}")
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _durable_remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Wiki transaction removal target must be a regular file: {path.name}")
    path.unlink()
    fsync_directory(path.parent)


def _remove_tree_durably(path: Path) -> None:
    parent = path.parent
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Wiki journal cleanup target must be a real directory")
        shutil.rmtree(path)
        fsync_directory(parent)


def fsync_directory(path: Path) -> None:
    """Flush directory metadata where the host exposes that durability boundary."""
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _required_hash(value: object) -> str:
    if not isinstance(value, str) or not value.startswith(_HASH_PREFIX):
        raise ValueError("Wiki journal hash is invalid")
    digest = value.removeprefix(_HASH_PREFIX)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Wiki journal hash is invalid")
    return value


def _optional_hash(value: object) -> str | None:
    return None if value is None else _required_hash(value)


def _strict_non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Wiki journal integer is invalid")
    return value


def _strict_identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Wiki journal transaction identifier is invalid")
    return value


def _hit_failpoint(_name: str) -> None:
    """Test seam for simulating a process interruption after durable boundaries."""
