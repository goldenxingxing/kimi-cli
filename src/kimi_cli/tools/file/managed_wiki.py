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
    if not isinstance(get_current_kaos(), LocalKaos):
        return _blocked_error()
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
    if not isinstance(get_current_kaos(), LocalKaos):
        raise ManagedWikiMutationBlocked("remote mutation cannot be verified")
    local_path = path.unsafe_to_local_path()
    root = manager.layout.root
    return await asyncio.to_thread(_write_verified_local, local_path, root, content, append)


def _write_verified_local(target: Path, root: Path, content: str, append: bool) -> int:
    """Write through a stable local file descriptor or fail closed."""
    if not hasattr(os, "O_NOFOLLOW") or os.name == "nt":
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
