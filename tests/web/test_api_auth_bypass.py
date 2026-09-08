"""A cookie has to be a session, not just a cookie.

``AuthMiddleware`` used to let any request carrying a ``kimi_session`` cookie
through, on the understanding that a per-route dependency would do the real
check. ``/api/config`` (which returns provider API keys), ``/api/sessions`` and
``/api/open-in`` have no such dependency, so on a deployment protected by
``KIMI_WEB_SESSION_TOKEN`` the header ``Cookie: kimi_session=anything`` was a
complete bypass of the bearer-token gate.

The WebSocket guard has always resolved the cookie against the session store
(``sessions._websocket_user``); this is the HTTP side catching up.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from kimi_cli.web.app import create_app


def _client() -> TestClient:
    return TestClient(create_app(session_token="the-session-token"))


def test_a_made_up_session_cookie_is_not_a_session() -> None:
    client = _client()
    client.cookies.set("kimi_session", "not-a-real-session-id")

    response = client.get("/api/config/")

    assert response.status_code == 401


def test_an_empty_bearer_and_no_cookie_is_still_rejected() -> None:
    client = _client()

    response = client.get("/api/config/")

    assert response.status_code == 401


def test_the_configured_bearer_token_still_works() -> None:
    """The bypass fix must not lock out the token it exists to enforce."""
    client = _client()

    response = client.get(
        "/api/config/",
        headers={"Authorization": "Bearer the-session-token"},
    )

    assert response.status_code == 200


def test_an_ordinary_user_cannot_read_the_provider_api_keys(monkeypatch) -> None:
    """config.toml is every provider's key in plaintext, and a PATCH here
    restarts everyone else's workers. Being logged in was the only check."""
    monkeypatch.setattr(
        "kimi_cli.web.user_auth.user_from_connection",
        lambda _c: {"id": "user-b", "role": "user"},
    )
    client = _client()
    client.cookies.set("kimi_session", "a-real-looking-session")

    assert client.get("/api/config/").status_code == 403
    assert client.get("/api/config/toml").status_code == 403


def test_an_admin_still_can(monkeypatch) -> None:
    monkeypatch.setattr(
        "kimi_cli.web.user_auth.user_from_connection",
        lambda _c: {"id": "root", "role": "admin"},
    )
    client = _client()
    client.cookies.set("kimi_session", "a-real-looking-session")

    assert client.get("/api/config/").status_code == 200
