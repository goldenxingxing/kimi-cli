from __future__ import annotations

import errno
import math
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from kimi_cli.wiki.locking import WikiBusyError, WikiLock


def _try_lock(path: str, mode: str, timeout: float, queue: multiprocessing.Queue[str]) -> None:
    lock = WikiLock(Path(path))
    try:
        context = lock.shared(timeout) if mode == "shared" else lock.exclusive(timeout)
        with context:
            queue.put("acquired")
    except WikiBusyError:
        queue.put("busy")


def test_exclusive_lock_times_out_while_another_process_holds_it(tmp_path: Path) -> None:
    lock_path = tmp_path / "locks" / "writer.lock"
    lock = WikiLock(lock_path)
    queue: multiprocessing.Queue[str] = multiprocessing.Queue()

    with lock.exclusive(1.0):
        process = multiprocessing.Process(
            target=_try_lock,
            args=(str(lock_path), "exclusive", 0.1, queue),
        )
        process.start()
        process.join(2)

    assert process.exitcode == 0
    assert queue.get(timeout=1) == "busy"


def test_shared_locks_coexist_but_block_a_writer(tmp_path: Path) -> None:
    lock_path = tmp_path / "locks" / "writer.lock"
    lock = WikiLock(lock_path)
    reader_queue: multiprocessing.Queue[str] = multiprocessing.Queue()
    writer_queue: multiprocessing.Queue[str] = multiprocessing.Queue()

    with lock.shared(1.0):
        reader = multiprocessing.Process(
            target=_try_lock,
            args=(str(lock_path), "shared", 0.5, reader_queue),
        )
        writer = multiprocessing.Process(
            target=_try_lock,
            args=(str(lock_path), "exclusive", 0.1, writer_queue),
        )
        reader.start()
        writer.start()
        reader.join(2)
        writer.join(2)

    assert reader.exitcode == 0
    assert writer.exitcode == 0
    expected_reader = "busy" if os.name == "nt" else "acquired"
    assert reader_queue.get(timeout=1) == expected_reader
    assert writer_queue.get(timeout=1) == "busy"


def test_lock_is_released_after_context_exit(tmp_path: Path) -> None:
    lock = WikiLock(tmp_path / "locks" / "writer.lock")

    with lock.exclusive(1.0):
        pass

    with lock.exclusive(0.1):
        assert True


def test_shared_lock_runs_recovery_before_exposing_data(tmp_path: Path) -> None:
    observed: list[str] = []
    lock = WikiLock(
        tmp_path / "locks" / "writer.lock",
        before_shared=lambda: observed.append("recovered"),
    )

    with lock.shared(1.0):
        observed.append("read")

    assert observed == ["recovered", "read"]


def test_lock_rejects_a_symlink_file(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.write_text("", encoding="utf-8")
    lock_path = tmp_path / "locks" / "writer.lock"
    lock_path.parent.mkdir()
    lock_path.symlink_to(target)

    with pytest.raises(ValueError, match="regular file"), WikiLock(lock_path).exclusive(0.1):
        pass


def test_negative_timeout_is_rejected(tmp_path: Path) -> None:
    with (
        pytest.raises(ValueError, match="non-negative"),
        WikiLock(tmp_path / "writer.lock").shared(-0.1),
    ):
        pass


@pytest.mark.parametrize("timeout", [False, math.nan, math.inf, -math.inf])
def test_non_finite_or_boolean_timeout_is_rejected(tmp_path: Path, timeout: float) -> None:
    with (
        pytest.raises(ValueError, match="finite non-negative"),
        WikiLock(tmp_path / "writer.lock").shared(timeout),
    ):
        pass


def test_timeout_is_deadline_based_not_attempt_count(tmp_path: Path) -> None:
    lock_path = tmp_path / "locks" / "writer.lock"
    started = time.monotonic()
    with (
        WikiLock(lock_path).exclusive(1.0),
        pytest.raises(WikiBusyError),
        WikiLock(lock_path).exclusive(0.05),
    ):
        pass
    assert time.monotonic() - started < 0.5


def test_failed_acquisition_does_not_attempt_windows_style_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kimi_cli.wiki.locking as locking_module

    def always_busy(*_args: object) -> None:
        raise OSError(errno.EAGAIN, "busy")

    def unexpected_unlock(*_args: object) -> None:
        raise AssertionError("unlock must only run after acquisition")

    monkeypatch.setattr(locking_module, "_try_lock", always_busy)
    monkeypatch.setattr(locking_module, "_unlock", unexpected_unlock)

    with (
        pytest.raises(WikiBusyError),
        WikiLock(tmp_path / "writer.lock").exclusive(0.0),
    ):
        pass


def test_unlock_error_after_acquisition_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kimi_cli.wiki.locking as locking_module

    def fail_unlock(*_args: object) -> None:
        raise OSError("unlock failed")

    monkeypatch.setattr(locking_module, "_unlock", fail_unlock)

    with (
        pytest.raises(OSError, match="unlock failed"),
        WikiLock(tmp_path / "writer.lock").exclusive(1.0),
    ):
        pass
