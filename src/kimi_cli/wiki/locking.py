"""Small cross-process shared/exclusive lock used by the global Wiki."""

from __future__ import annotations

import errno
import os
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import BinaryIO, Literal


class WikiBusyError(TimeoutError):
    """Raised when the Wiki lock cannot be acquired before its deadline."""


class WikiLock:
    """A deadline-based advisory lock backed by one regular sidecar file.

    ``before_shared`` runs while an exclusive lock is held.  The lock is then
    downgraded to shared without closing the file descriptor, so recovery can
    finish before any reader observes authoritative Markdown.
    """

    def __init__(
        self,
        path: Path,
        *,
        before_shared: Callable[[], object] | None = None,
    ) -> None:
        self.path = path
        self._before_shared = before_shared

    @contextmanager
    def shared(self, timeout: float) -> Iterator[None]:
        """Acquire a shared lock, running configured recovery first."""
        self._validate_timeout(timeout)
        stream = self._open_lock_file()
        try:
            initial_mode: Literal["shared", "exclusive"] = (
                "exclusive" if self._before_shared is not None else "shared"
            )
            self._acquire(stream, initial_mode, timeout)
            if self._before_shared is not None:
                self._before_shared()
                self._convert_to_shared(stream)
            yield
        finally:
            self._unlock_and_close(stream)

    @contextmanager
    def exclusive(self, timeout: float) -> Iterator[None]:
        """Acquire the exclusive writer lock until the context exits."""
        self._validate_timeout(timeout)
        stream = self._open_lock_file()
        try:
            self._acquire(stream, "exclusive", timeout)
            yield
        finally:
            self._unlock_and_close(stream)

    @staticmethod
    def _validate_timeout(timeout: float) -> None:
        if timeout < 0:
            raise ValueError("Wiki lock timeout must be non-negative")

    def _open_lock_file(self) -> BinaryIO:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"Wiki lock parent must be a real directory: {parent}")
        if self.path.is_symlink():
            raise ValueError(f"Wiki lock path must be a regular file: {self.path}")

        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise ValueError(f"unable to open Wiki lock file: {self.path}") from exc
        try:
            file_stat = os.fstat(descriptor)
            path_stat = self.path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or not stat.S_ISREG(path_stat.st_mode)
                or (file_stat.st_dev, file_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise ValueError(f"Wiki lock path must be a regular file: {self.path}")
            if os.name == "nt" and file_stat.st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            return os.fdopen(descriptor, "r+b", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise

    def _acquire(
        self,
        stream: BinaryIO,
        mode: Literal["shared", "exclusive"],
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                _try_lock(stream, mode)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WikiBusyError(f"Wiki is busy; timed out acquiring {mode} lock") from exc
                time.sleep(min(0.01, remaining))

    @staticmethod
    def _convert_to_shared(stream: BinaryIO) -> None:
        if os.name == "nt":
            # msvcrt exposes no shared byte-range lock. Keeping the exclusive
            # lock is conservative and preserves correctness on Windows.
            return
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)

    @staticmethod
    def _unlock_and_close(stream: BinaryIO) -> None:
        try:
            _unlock(stream)
        finally:
            stream.close()


def _try_lock(stream: BinaryIO, mode: Literal["shared", "exclusive"]) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    operation = fcntl.LOCK_SH if mode == "shared" else fcntl.LOCK_EX
    fcntl.flock(stream.fileno(), operation | fcntl.LOCK_NB)


def _unlock(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        # The acquisition path may have raised before the byte was locked.
        with suppress(OSError):
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
