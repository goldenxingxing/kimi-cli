"""Applying a settings change to a live server instead of restarting it.

Two things make "restart nothing, interrupt nobody" actually work, and both are
easy to get subtly wrong: the server has to refresh its *own* environment (it
was frozen at spawn, and a stale copy resurrects providers the user deleted),
and a session that was busy when the change landed has to pick it up later
rather than never.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kimi_cli.web.config_reload import (
    ENV_FILE_VAR,
    RELOADABLE_KEYS,
    parse_env_file,
    refresh_environment,
)
from kimi_cli.web.runner.process import KimiCLIRunner, SessionProcess


@pytest.fixture(autouse=True)
def _no_ambient_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's own shell must not decide what these tests observe."""
    for key in RELOADABLE_KEYS:
        monkeypatch.delenv(key, raising=False)


def _write_env(tmp_path, body: str, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_VAR, str(env_file))


def _live(proc: SessionProcess) -> SessionProcess:
    """Give the session a worker that is running, without spawning one."""
    proc._process = SimpleNamespace(returncode=None)  # type: ignore[assignment]  # noqa: SLF001
    return proc


def test_parses_the_lines_the_desktop_app_writes() -> None:
    parsed = parse_env_file(
        '# a comment\nLLM_PROVIDERS=[{"name":"kimi"}]\nKIMI_BASE_URL="https://x/v1"\n\nBROKEN\n'
    )

    assert parsed == {
        "LLM_PROVIDERS": '[{"name":"kimi"}]',
        "KIMI_BASE_URL": "https://x/v1",
    }


def test_a_changed_provider_reaches_this_process(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDERS", '[{"name":"kimi","api_key":"old"}]')
    _write_env(tmp_path, 'LLM_PROVIDERS=[{"name":"kimi","api_key":"new"}]\n', monkeypatch)

    changed = refresh_environment()

    assert changed == {"LLM_PROVIDERS"}
    assert os.environ["LLM_PROVIDERS"] == '[{"name":"kimi","api_key":"new"}]'


def test_a_deleted_provider_is_removed_not_merely_left_behind(tmp_path, monkeypatch) -> None:
    """The whole reason the env is refreshed at all.

    `_build_global_config` merges env providers back into `config.toml` on every
    request, so a key left in place would write the deleted provider back to
    disk and hand it to every worker spawned afterwards. Deleting the last
    provider drops the line from `.env` entirely (`write_env` removes a key set
    to ""), so absence -- not an empty value -- is what has to be honoured.
    """
    monkeypatch.setenv("LLM_PROVIDERS", '[{"name":"kimi"},{"name":"gone"}]')
    _write_env(tmp_path, "KIMI_BASE_URL=https://x/v1\n", monkeypatch)

    changed = refresh_environment()

    assert "LLM_PROVIDERS" in changed
    assert "LLM_PROVIDERS" not in os.environ


def test_a_settings_file_cannot_rewrite_arbitrary_variables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH_SENTINEL", "untouched")
    _write_env(tmp_path, "PATH_SENTINEL=hijacked\nKIMI_WEB_PORT=9999\n", monkeypatch)
    monkeypatch.setenv("KIMI_WEB_PORT", "5494")

    changed = refresh_environment()

    assert changed == set()
    assert os.environ["PATH_SENTINEL"] == "untouched"
    # The port is bound; claiming to reload it would be a lie.
    assert os.environ["KIMI_WEB_PORT"] == "5494"


def test_no_env_file_means_nothing_to_refresh(monkeypatch) -> None:
    monkeypatch.delenv(ENV_FILE_VAR, raising=False)

    assert refresh_environment() == set()


@pytest.mark.asyncio
async def test_a_busy_session_is_deferred_not_dropped() -> None:
    """A prompt in flight must not be interrupted, and must not be forgotten."""
    runner = KimiCLIRunner()
    session_id = uuid4()
    proc = _live(SessionProcess(session_id))
    proc._in_flight_prompt_ids.add("prompt-1")  # noqa: SLF001 - simulating a live prompt
    restarted: list[str | None] = []

    async def _record(*, reason: str | None = None) -> None:
        restarted.append(reason)

    proc.restart_worker = _record  # type: ignore[method-assign]
    runner._sessions[session_id] = proc  # noqa: SLF001 - no worker to spawn in a test

    summary = await runner.restart_running_workers(reason="config_reload", force=False)

    assert summary.restarted_session_ids == []
    assert summary.skipped_busy_session_ids == [session_id]
    assert restarted == []
    assert proc._deferred_restart_reason == "config_reload"  # noqa: SLF001


@pytest.mark.asyncio
async def test_the_deferred_restart_lands_when_the_prompt_finishes() -> None:
    session_id = uuid4()
    proc = _live(SessionProcess(session_id))
    proc._in_flight_prompt_ids.add("prompt-1")  # noqa: SLF001
    restarted: list[str | None] = []

    async def _record(*, reason: str | None = None) -> None:
        restarted.append(reason)

    proc.restart_worker = _record  # type: ignore[method-assign]
    proc.defer_restart("config_reload")

    # Still busy: nothing happens yet.
    proc._settle_deferred_restart()  # noqa: SLF001
    await asyncio.sleep(0)
    assert restarted == []

    # The prompt finishes.
    proc._in_flight_prompt_ids.clear()  # noqa: SLF001
    proc._settle_deferred_restart()  # noqa: SLF001
    assert proc._deferred_restart_task is not None  # noqa: SLF001
    await proc._deferred_restart_task  # noqa: SLF001

    assert restarted == ["config_reload"]


@pytest.mark.asyncio
async def test_touching_the_signal_applies_the_change(tmp_path, monkeypatch) -> None:
    """The whole loop: the desktop app touches a file, sessions pick the change up.

    Every other test here covers one half. This is the part that silently does
    nothing if the poll loop is wrong -- and silently doing nothing is exactly
    the failure the reload exists to replace.
    """
    from types import SimpleNamespace

    from kimi_cli.web import config_reload

    monkeypatch.setenv(config_reload.APP_DATA_DIR_VAR, str(tmp_path))
    monkeypatch.setattr(config_reload, "POLL_INTERVAL_S", 0.01)
    _write_env(tmp_path, 'LLM_PROVIDERS=[{"name":"added"}]\n', monkeypatch)

    applied = asyncio.Event()

    class _Runner:
        async def restart_running_workers(self, *, reason: str, force: bool):
            assert force is False, "a reload must never interrupt a busy session"
            applied.set()
            return SimpleNamespace(restarted_session_ids=[], skipped_busy_session_ids=[])

    app = SimpleNamespace(state=SimpleNamespace())
    watcher = asyncio.create_task(config_reload.watch_for_reload(app, _Runner()))
    try:
        await asyncio.sleep(0.05)
        assert not applied.is_set(), "nothing was requested yet"

        (tmp_path / config_reload.RELOAD_SIGNAL_NAME).write_text("1", encoding="utf-8")
        await asyncio.wait_for(applied.wait(), timeout=2.0)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

    assert os.environ["LLM_PROVIDERS"] == '[{"name":"added"}]'
