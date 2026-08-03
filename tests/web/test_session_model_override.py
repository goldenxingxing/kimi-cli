"""Tests for per-session model override (session_config.json ``model`` key)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest
from kaos.path import KaosPath

from kimi_cli.session import Session as KimiCLISession
from kimi_cli.web.api import sessions as sessions_api
from kimi_cli.web.api.sessions import CreateSessionRequest
from kimi_cli.web.models import UpdateSessionRequest
from kimi_cli.web.runner.process import KimiCLIRunner

if TYPE_CHECKING:
    from fastapi import Request

TEST_MODEL = "test-model-a"
OTHER_MODEL = "test-model-b"


@pytest.fixture
def isolated_share_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    share_dir = tmp_path / "share"
    share_dir.mkdir()

    def _get_share_dir() -> Path:
        share_dir.mkdir(parents=True, exist_ok=True)
        return share_dir

    monkeypatch.setattr("kimi_cli.share.get_share_dir", _get_share_dir)
    monkeypatch.setattr("kimi_cli.metadata.get_share_dir", _get_share_dir)
    return share_dir


@pytest.fixture
def work_dir(tmp_path: Path) -> KaosPath:
    path = tmp_path / "work"
    path.mkdir()
    return KaosPath.unsafe_from_local_path(path)


@pytest.fixture
def known_models(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    models = {TEST_MODEL, OTHER_MODEL}
    monkeypatch.setattr(
        "kimi_cli.web.api.config.get_effective_model_names",
        lambda: models,
    )
    return models


def _fake_http_request() -> Request:
    return cast("Request", SimpleNamespace(cookies={}, headers={}))


class _FakeSessionProcess:
    """Stand-in for ``SessionProcess`` recording restarts."""

    def __init__(self, *, is_busy: bool = False, is_running: bool = False) -> None:
        self.is_busy = is_busy
        self.is_running = is_running
        self.restart_reasons: list[str | None] = []

    async def restart_worker(self, *, reason: str | None = None) -> None:
        self.restart_reasons.append(reason)


class _FakeRunner:
    """Stand-in for ``KimiCLIRunner`` for tests that bypass FastAPI DI."""

    def __init__(self, process: _FakeSessionProcess | None = None) -> None:
        self._process = process

    def get_session(self, _session_id: UUID) -> _FakeSessionProcess | None:
        return self._process


@pytest.mark.anyio
async def test_create_session_with_model_persists_override(
    isolated_share_dir: Path,
    work_dir: KaosPath,
    known_models: set[str],
) -> None:
    result = await sessions_api.create_session(
        _fake_http_request(),
        CreateSessionRequest(work_dir=str(work_dir), model=TEST_MODEL),
    )

    assert result.model == TEST_MODEL
    assert result.session_dir is not None
    cfg = json.loads((Path(result.session_dir) / "session_config.json").read_text())
    assert cfg["model"] == TEST_MODEL


@pytest.mark.anyio
async def test_create_session_rejects_unknown_model(
    isolated_share_dir: Path,
    work_dir: KaosPath,
    known_models: set[str],
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await sessions_api.create_session(
            _fake_http_request(),
            CreateSessionRequest(work_dir=str(work_dir), model="no-such-model"),
        )
    assert exc_info.value.status_code == 400


@pytest.mark.anyio
async def test_create_session_without_model_has_no_override(
    isolated_share_dir: Path,
    work_dir: KaosPath,
) -> None:
    result = await sessions_api.create_session(
        _fake_http_request(),
        CreateSessionRequest(work_dir=str(work_dir)),
    )

    assert result.model is None
    assert result.session_dir is not None
    assert not (Path(result.session_dir) / "session_config.json").exists()


@pytest.mark.anyio
async def test_update_session_model_persists_and_preserves_thinking(
    isolated_share_dir: Path,
    work_dir: KaosPath,
    known_models: set[str],
) -> None:
    session = await KimiCLISession.create(work_dir)
    # Pre-existing per-session config that must survive the model update.
    (session.dir / "session_config.json").write_text(
        json.dumps({"thinking": True}), encoding="utf-8"
    )

    runner = cast(KimiCLIRunner, _FakeRunner())
    updated = await sessions_api.update_session(
        UUID(session.id),
        UpdateSessionRequest(model=OTHER_MODEL),
        runner=runner,
    )

    assert updated.model == OTHER_MODEL
    cfg = json.loads((session.dir / "session_config.json").read_text())
    assert cfg["model"] == OTHER_MODEL
    assert cfg["thinking"] is True


@pytest.mark.anyio
async def test_update_session_model_restarts_only_own_worker(
    isolated_share_dir: Path,
    work_dir: KaosPath,
    known_models: set[str],
) -> None:
    session = await KimiCLISession.create(work_dir)
    process = _FakeSessionProcess(is_busy=False, is_running=True)
    runner = cast(KimiCLIRunner, _FakeRunner(process))

    await sessions_api.update_session(
        UUID(session.id),
        UpdateSessionRequest(model=TEST_MODEL),
        runner=runner,
    )

    assert process.restart_reasons == ["model_update"]


@pytest.mark.anyio
async def test_update_session_model_rejected_when_busy(
    isolated_share_dir: Path,
    work_dir: KaosPath,
    known_models: set[str],
) -> None:
    from fastapi import HTTPException

    session = await KimiCLISession.create(work_dir)
    process = _FakeSessionProcess(is_busy=True, is_running=True)
    runner = cast(KimiCLIRunner, _FakeRunner(process))

    with pytest.raises(HTTPException) as exc_info:
        await sessions_api.update_session(
            UUID(session.id),
            UpdateSessionRequest(model=TEST_MODEL),
            runner=runner,
        )
    assert exc_info.value.status_code == 409
    assert not (session.dir / "session_config.json").exists()
    assert process.restart_reasons == []


@pytest.mark.anyio
async def test_update_session_model_rejects_unknown_model(
    isolated_share_dir: Path,
    work_dir: KaosPath,
    known_models: set[str],
) -> None:
    from fastapi import HTTPException

    session = await KimiCLISession.create(work_dir)
    runner = cast(KimiCLIRunner, _FakeRunner())

    with pytest.raises(HTTPException) as exc_info:
        await sessions_api.update_session(
            UUID(session.id),
            UpdateSessionRequest(model="no-such-model"),
            runner=runner,
        )
    assert exc_info.value.status_code == 400


def test_load_session_by_id_reads_model_override(
    isolated_share_dir: Path,
    work_dir: KaosPath,
) -> None:
    from kimi_cli.metadata import Metadata, WorkDirMeta, save_metadata
    from kimi_cli.web.store.sessions import invalidate_sessions_cache, load_session_by_id

    session_id = UUID("12345678-1234-5678-1234-567812345678")
    work_meta = WorkDirMeta(path=str(work_dir))
    session_dir = work_meta.legacy_sessions_dir / str(session_id)
    session_dir.mkdir(parents=True)
    (session_dir / "context.jsonl").write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")
    save_metadata(Metadata(work_dirs=[work_meta]))

    invalidate_sessions_cache()
    loaded = load_session_by_id(session_id)
    assert loaded is not None
    assert loaded.model is None

    (session_dir / "session_config.json").write_text(
        json.dumps({"model": TEST_MODEL}), encoding="utf-8"
    )
    invalidate_sessions_cache()
    loaded = load_session_by_id(session_id)
    assert loaded is not None
    assert loaded.model == TEST_MODEL

    # Corrupt config file → treated as no override.
    (session_dir / "session_config.json").write_text("not-json", encoding="utf-8")
    invalidate_sessions_cache()
    loaded = load_session_by_id(session_id)
    assert loaded is not None
    assert loaded.model is None


def test_worker_read_session_overrides(tmp_path: Path) -> None:
    from kimi_cli.web.runner.worker import read_session_overrides

    assert read_session_overrides(tmp_path) == {}

    cfg_file = tmp_path / "session_config.json"
    cfg_file.write_text(json.dumps({"thinking": True, "model": TEST_MODEL}), encoding="utf-8")
    cfg = read_session_overrides(tmp_path)
    assert cfg["thinking"] is True
    assert cfg["model"] == TEST_MODEL

    cfg_file.write_text("not-json", encoding="utf-8")
    assert read_session_overrides(tmp_path) == {}


@pytest.mark.anyio
async def test_restart_running_workers_skips_model_override_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kimi_cli.web.runner import process as process_mod

    override_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    plain_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    restarted: list[str | None] = []

    class _Proc:
        is_running = True
        is_busy = False

        def __init__(self, sid: UUID) -> None:
            self._sid = sid

        async def restart_worker(self, *, reason: str | None = None) -> None:
            restarted.append(f"{self._sid}:{reason}")

    runner = KimiCLIRunner()
    runner._sessions[override_id] = cast(process_mod.SessionProcess, _Proc(override_id))
    runner._sessions[plain_id] = cast(process_mod.SessionProcess, _Proc(plain_id))

    def fake_load_session_by_id(session_id: UUID) -> SimpleNamespace | None:
        if session_id == override_id:
            return SimpleNamespace(model=TEST_MODEL)
        return SimpleNamespace(model=None)

    monkeypatch.setattr(process_mod, "load_session_by_id", fake_load_session_by_id)

    summary = await runner.restart_running_workers(
        reason="config_update",
        force=False,
        skip_model_override=True,
    )

    assert summary.restarted_session_ids == [plain_id]
    assert summary.skipped_busy_session_ids == []
    assert restarted == [f"{plain_id}:config_update"]

    # Without the skip flag both sessions restart (existing behaviour).
    restarted.clear()
    summary = await runner.restart_running_workers(reason="config_update", force=False)
    assert set(summary.restarted_session_ids) == {override_id, plain_id}
