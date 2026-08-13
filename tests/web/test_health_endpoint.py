"""The /healthz contract the desktop supervisor depends on."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from kimi_cli.web.app import create_app


def test_health_reports_the_serving_process_id() -> None:
    """The pid is what lets a supervisor tell its own server from a stranger's.

    Without it, a second copy of the app reads someone else's 200 as proof
    that its own child came up, and respawns a child that dies on a port
    conflict every time — silently, forever.
    """
    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pid"] == os.getpid()
