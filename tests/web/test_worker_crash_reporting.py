"""The worker must get its own crash traceback out before the process dies.

`enable_logging` redirects the worker's fd=2 into loguru, drained by a daemon
thread the interpreter kills at shutdown. A traceback printed by the default
excepthook therefore raced that shutdown and usually arrived truncated to its
first line — on the pipe the parent reads and in the log alike. These tests
spawn a real worker that fails after logging is enabled and check both.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def crashed_worker(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a worker against a session id that does not exist.

    `run_worker` raises `ValueError` for it, after `enable_logging` has already
    swapped fd=2 — the shape of every crash this reporting exists for.
    """
    env = dict(os.environ)
    env["KIMI_SHARE_DIR"] = str(tmp_path)
    env["KIMI_SESSION_DATA_DIR"] = str(tmp_path / "sessions")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "kimi_cli.web.runner.worker", str(uuid4())],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )


def test_crash_traceback_reaches_the_parent(
    crashed_worker: subprocess.CompletedProcess[str],
) -> None:
    """SessionProcess reads this pipe and shows it in the UI, so the whole
    traceback has to be on it — not just its first line."""
    assert crashed_worker.returncode == 1
    assert "Traceback (most recent call last):" in crashed_worker.stderr
    assert "ValueError: Session not found" in crashed_worker.stderr


def test_crash_traceback_reaches_the_log(
    crashed_worker: subprocess.CompletedProcess[str],
    tmp_path: Path,
) -> None:
    """And it is in the log too, for anyone reading it after the fact."""
    log = (tmp_path / "logs" / "kimi.log").read_text(encoding="utf-8")
    assert "Worker crashed:" in log
    assert "ValueError: Session not found" in log
