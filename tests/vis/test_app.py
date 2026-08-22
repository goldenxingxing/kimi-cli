from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from kimi_cli.metadata import Metadata, WorkDirMeta, save_metadata
from kimi_cli.vis.api import system as vis_system_api
from kimi_cli.vis.app import create_app


def test_vis_sessions_include_session_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))

    work_dir = tmp_path / "project"
    work_dir.mkdir()
    metadata = Metadata(work_dirs=[WorkDirMeta(path=str(work_dir))])
    save_metadata(metadata)

    session_dir = metadata.work_dirs[0].sessions_dir / "session123"
    session_dir.mkdir(parents=True)
    (session_dir / "context.jsonl").write_text("{}\n", encoding="utf-8")

    with TestClient(create_app()) as client:
        response = client.get("/api/vis/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["session_id"] == "session123"
    assert payload[0]["session_dir"] == str(session_dir)
    assert payload[0]["work_dir"] == str(work_dir)


def test_vis_app_mounts_open_in_route() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/open-in",
            json={"app": "finder", "path": "/definitely/missing/path"},
        )

    assert response.status_code == 400


def test_vis_capabilities_report_open_in_support(monkeypatch) -> None:
    monkeypatch.setattr(vis_system_api.sys, "platform", "linux")

    with TestClient(create_app()) as client:
        response = client.get("/api/vis/capabilities")

    assert response.status_code == 200
    assert response.json() == {"open_in_supported": False}


def test_vis_sessions_cover_both_storage_locations(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Sessions moved into the work directory; the share directory holds the old ones.

    The listing scanned only the share directory for as long as the move had
    been in place, so every session written since was invisible — and the page
    still rendered, showing a shorter history rather than an error. Nothing
    covered the statistics endpoint at all, which had the same scan.
    """
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))

    work_dir = tmp_path / "project"
    work_dir.mkdir()
    metadata = Metadata(work_dirs=[WorkDirMeta(path=str(work_dir))])
    save_metadata(metadata)
    work = metadata.work_dirs[0]

    current = work.sessions_dir / "current-session"
    current.mkdir(parents=True)
    (current / "context.jsonl").write_text("{}\n", encoding="utf-8")

    legacy = work.legacy_sessions_dir / "legacy-session"
    legacy.mkdir(parents=True)
    (legacy / "context.jsonl").write_text("{}\n", encoding="utf-8")

    with TestClient(create_app()) as client:
        payload = client.get("/api/vis/sessions").json()

    assert {row["session_id"] for row in payload} == {"current-session", "legacy-session"}
    assert all(row["work_dir"] == str(work_dir) for row in payload)
