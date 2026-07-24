"""Shared protection against bypassing the managed Wiki mutation boundary."""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, cast

from kaos import get_current_kaos
from kaos.local import LocalKaos
from kosong.tooling import ToolError

if TYPE_CHECKING:
    from kaos.path import KaosPath

    from kimi_cli.soul.agent import Runtime
    from kimi_cli.wiki.manager import WikiManager


class ManagedWikiMutationBlocked(RuntimeError):
    """A file-tool mutation cannot prove it remains outside the managed Wiki."""


def reject_managed_wiki_target(path: KaosPath, runtime: Runtime) -> ToolError | None:
    """Reject any local descendant or symlink alias of the managed Wiki root.

    The Wiki root is intentionally not an ordinary workspace.  This check occurs
    before diff rendering or approval so no file-tool code can obtain a generic
    filesystem approval for a managed Wiki mutation.
    """
    manager = cast("WikiManager | None", getattr(runtime, "wiki", None))
    if manager is None:
        return None
    if not _is_local_kaos():
        return _blocked_error() if _is_remote_managed_target(path, runtime, manager) else None
    try:
        target = Path(str(path)).expanduser().resolve(strict=False)
        root = manager.layout.root.resolve(strict=True)
        if target.is_relative_to(root):
            return ToolError(
                message=_blocked_error().message,
                brief="Use Wiki tool",
            )
    except OSError:
        # Let the normal file tool surface path/remote filesystem errors.  A path
        # we cannot prove is local must not be accidentally treated as the root.
        return None
    return None


async def write_verified_text(
    path: KaosPath,
    runtime: Runtime,
    content: str,
    *,
    append: bool = False,
) -> int:
    """Mutate a local file only after final no-follow/inode verification.

    The first target check is only a fast pre-approval rejection.  This function
    is the authoritative boundary: it opens the final path through a no-follow
    parent descriptor, rejects links, and writes through that descriptor rather
    than resolving the mutable pathname a second time.
    """
    manager = cast("WikiManager | None", getattr(runtime, "wiki", None))
    if manager is None:
        return await (path.append_text(content) if append else path.write_text(content))
    if not _is_local_kaos():
        if _is_remote_managed_target(path, runtime, manager):
            raise ManagedWikiMutationBlocked("remote target maps to managed Wiki")
        # A remote filesystem cannot alias a local Wiki unless deployment wiring
        # explicitly maps a managed root. Do not expose local root paths remotely.
        return await (path.append_text(content) if append else path.write_text(content))
    local_path = path.unsafe_to_local_path()
    root = manager.layout.root
    if _is_windows():
        return await asyncio.to_thread(_write_verified_windows, local_path, root, content, append)
    return await asyncio.to_thread(_write_verified_local, local_path, root, content, append)


def _write_verified_local(target: Path, root: Path, content: str, append: bool) -> int:
    """Write through a stable local file descriptor or fail closed."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise ManagedWikiMutationBlocked("secure no-follow writes are unavailable")
    root = root.resolve(strict=True)
    resolved = target.resolve(strict=False)
    if resolved.is_relative_to(root):
        raise ManagedWikiMutationBlocked("target resolves inside managed Wiki")
    try:
        parent = target.parent.resolve(strict=True)
    except OSError as exc:
        raise ManagedWikiMutationBlocked("target parent cannot be verified") from exc
    if parent.is_relative_to(root):
        raise ManagedWikiMutationBlocked("target parent resolves inside managed Wiki")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        parent_fd = os.open(parent, parent_flags)
    except OSError as exc:
        raise ManagedWikiMutationBlocked("target parent cannot be opened safely") from exc
    try:
        fd = _open_verified_target(parent_fd, target.name)
        try:
            _write_fd(fd, content.encode("utf-8"), append=append)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    return len(content)


def _write_verified_windows(target: Path, root: Path, content: str, append: bool) -> int:
    """Verify the final Windows leaf identity before writing through its handle.

    Python does not expose POSIX ``O_NOFOLLOW`` semantics on Windows. We therefore
    resolve the final target, reject symlinks and every multiply-linked inode, then
    compare the lstat and opened-handle identities before writing. A direct or
    aliased managed-root target is rejected; ordinary one-link workspace files
    remain writable.
    """
    root = root.resolve(strict=True)
    resolved = target.resolve(strict=False)
    if resolved.is_relative_to(root):
        raise ManagedWikiMutationBlocked("target resolves inside managed Wiki")
    try:
        parent = target.parent.resolve(strict=True)
    except OSError as exc:
        raise ManagedWikiMutationBlocked("target parent cannot be verified") from exc
    if parent.is_relative_to(root):
        raise ManagedWikiMutationBlocked("target parent resolves inside managed Wiki")
    try:
        before = os.lstat(target)
    except FileNotFoundError:
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except OSError as exc:
            raise ManagedWikiMutationBlocked("new target cannot be opened safely") from exc
    else:
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1:
            raise ManagedWikiMutationBlocked("target is linked or not a regular file")
        try:
            fd = os.open(target, os.O_WRONLY)
        except OSError as exc:
            raise ManagedWikiMutationBlocked("target cannot be opened safely") from exc
        after = os.fstat(fd)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
        ):
            os.close(fd)
            raise ManagedWikiMutationBlocked("target changed during verification")
    try:
        actual = os.fstat(fd)
        if not stat.S_ISREG(actual.st_mode) or actual.st_nlink != 1:
            raise ManagedWikiMutationBlocked("target is linked or not a regular file")
        _write_fd(fd, content.encode("utf-8"), append=append)
    finally:
        os.close(fd)
    return len(content)


def _open_verified_target(parent_fd: int, name: str) -> int:
    """Open a regular, single-link final inode without following symlinks."""
    flags = os.O_WRONLY | os.O_NOFOLLOW
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            fd = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=parent_fd)
        except OSError as exc:
            raise ManagedWikiMutationBlocked("new target cannot be opened safely") from exc
    else:
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise ManagedWikiMutationBlocked("target is not a regular file")
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ManagedWikiMutationBlocked("target cannot be opened safely") from exc
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            os.close(fd)
            raise ManagedWikiMutationBlocked("target changed during verification")
    actual = os.fstat(fd)
    if not stat.S_ISREG(actual.st_mode) or actual.st_nlink != 1:
        os.close(fd)
        raise ManagedWikiMutationBlocked("target is linked or not a regular file")
    return fd


def _write_fd(fd: int, data: bytes, *, append: bool) -> None:
    if not append:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
    else:
        os.lseek(fd, 0, os.SEEK_END)
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        view = view[count:]
    os.fsync(fd)


def _blocked_error() -> ToolError:
    return ToolError(
        message="Managed Wiki files can only be changed through the Wiki tool.",
        brief="Use Wiki tool",
    )


def _is_local_kaos() -> bool:
    return isinstance(get_current_kaos(), LocalKaos)


def _is_windows() -> bool:
    return os.name == "nt"


def _is_remote_managed_target(path: KaosPath, runtime: Runtime, manager: WikiManager) -> bool:
    """Use only explicitly expressible remote root mappings; never resolve locally."""
    candidate = str(path.canonical())
    configured = cast("object", getattr(runtime, "wiki_remote_roots", ()))
    roots = [str(manager.layout.root)]
    if isinstance(configured, (list, tuple, set, frozenset)):
        configured_roots = cast(
            "list[object] | tuple[object, ...] | set[object] | frozenset[object]", configured
        )
        for configured_root in configured_roots:
            if isinstance(configured_root, str):
                roots.append(configured_root)
    return any(_is_descendant_text(candidate, root) for root in roots)


def _is_descendant_text(path: str, root: str) -> bool:
    normalized_path = path.rstrip("/\\")
    normalized_root = root.rstrip("/\\")
    return normalized_path == normalized_root or normalized_path.startswith(
        (normalized_root + "/", normalized_root + "\\")
    )
