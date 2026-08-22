"""The session socket must accept whatever the HTTP API accepts.

An install with ``KIMI_WEB_SESSION_TOKEN`` set and a user logged in through
the login page used to get 200 on every API call — ``AuthMiddleware`` falls
back to the ``kimi_session`` cookie — and a rejected handshake on every
socket, because the WebSocket guard only ever looked at ``?token=``. The UI
has no way to render that except "lost connection to the session and could
not reconnect", so it retried five times and gave up.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from kimi_cli.web.api import sessions


@pytest.fixture
def app_with_token(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """A minimal app carrying just the guard under test."""
    app = FastAPI()
    app.state.session_token = "the-session-token"
    app.state.enforce_origin = False
    app.state.allowed_origins = []
    app.state.lan_only = False

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        expected_token = websocket.app.state.session_token
        token = websocket.query_params.get("token")
        if not sessions.verify_token(token, expected_token) and not sessions._websocket_user(
            websocket
        ):
            await websocket.close(code=4401, reason="Auth required")
            return
        await websocket.accept()
        await websocket.send_text("ok")
        await websocket.close()

    return app


def _logged_in(monkeypatch: pytest.MonkeyPatch, user: dict[str, Any] | None) -> None:
    monkeypatch.setattr("kimi_cli.web.user_auth.user_from_connection", lambda connection: user)


def test_session_token_in_the_query_still_works(
    app_with_token: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _logged_in(monkeypatch, None)
    with (
        TestClient(app_with_token) as client,
        client.websocket_connect("/stream?token=the-session-token") as ws,
    ):
        assert ws.receive_text() == "ok"


def test_a_logged_in_browser_is_accepted_without_the_session_token(
    app_with_token: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this file exists for."""
    _logged_in(monkeypatch, {"id": "u1", "username": "someone"})
    with TestClient(app_with_token) as client, client.websocket_connect("/stream") as ws:
        assert ws.receive_text() == "ok"


def test_neither_credential_is_still_refused(
    app_with_token: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widening the door must not remove it."""
    _logged_in(monkeypatch, None)
    with (
        TestClient(app_with_token) as client,
        pytest.raises(WebSocketDisconnect) as excinfo,
        client.websocket_connect("/stream?token=wrong") as ws,
    ):
        ws.receive_text()
    assert excinfo.value.code == 4401


def test_a_failing_user_lookup_does_not_break_the_handshake(
    app_with_token: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database hiccup must refuse the socket, not raise inside the guard."""

    def boom(connection):
        raise RuntimeError("db is down")

    monkeypatch.setattr("kimi_cli.web.user_auth.user_from_connection", boom)
    with (
        TestClient(app_with_token) as client,
        pytest.raises(WebSocketDisconnect) as excinfo,
        client.websocket_connect("/stream") as ws,
    ):
        ws.receive_text()
    assert excinfo.value.code == 4401
