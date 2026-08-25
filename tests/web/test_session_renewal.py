"""A session in use must not expire out from under the person using it.

Before renewal existed, ``expires_at`` was stamped once at login and never
moved: a tab open across the 24-hour mark got a login page mid-use, and the
only way to find out was to refresh. These tests pin the three properties
that fix depends on — it slides while you are active, it stays put while
there is plenty of time left, and it stops dead at the absolute ceiling.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kimi_cli.web import session_policy
from kimi_cli.web.db.crud import get_user_session_row
from kimi_cli.web.db.database import get_db, init_db


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """An app on a throwaway users.db, with the default admin seeded."""
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    monkeypatch.setenv("KIMI_USE_CONTAINERS", "false")
    monkeypatch.delenv("KIMI_WEB_SESSION_TOKEN", raising=False)
    init_db()

    from kimi_cli.web.app import create_app

    with TestClient(create_app()) as c:
        resp = c.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        assert resp.status_code == 200, resp.text
        yield c


def _token(client: TestClient) -> str:
    token = client.cookies.get(session_policy.COOKIE_NAME)
    assert token
    return token


def _row(token: str) -> dict:
    db = get_db()
    try:
        row = get_user_session_row(db, token)
    finally:
        db.close()
    assert row is not None
    return row


def _set(token: str, *, created_at: float | None = None, expires_at: float | None = None) -> None:
    db = get_db()
    try:
        if created_at is not None:
            db.execute(
                "UPDATE user_sessions SET created_at = ? WHERE token = ?", (created_at, token)
            )
        if expires_at is not None:
            db.execute(
                "UPDATE user_sessions SET expires_at = ? WHERE token = ?", (expires_at, token)
            )
        db.commit()
    finally:
        db.close()


def test_a_fresh_session_is_left_alone(client: TestClient) -> None:
    """Renewing on every request would write the database for nothing."""
    token = _token(client)
    before = _row(token)["expires_at"]

    resp = client.get("/api/auth/me")

    assert resp.status_code == 200
    assert "set-cookie" not in {k.lower() for k in resp.headers}
    assert _row(token)["expires_at"] == before


def test_a_session_past_half_its_window_slides_forward(client: TestClient) -> None:
    """The whole point: keep using it and the deadline keeps moving."""
    token = _token(client)
    now = time.time()
    _set(token, expires_at=now + 60)  # an hour of a 24h window would do; a minute is clearer

    resp = client.get("/api/auth/me")

    assert resp.status_code == 200
    assert any(k.lower() == "set-cookie" for k in resp.headers)
    assert _row(token)["expires_at"] > now + session_policy.session_max_age() - 30


def test_renewal_stops_at_the_absolute_ceiling(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cookie that is used forever still dies at the line drawn from login."""
    monkeypatch.setenv("KIMI_WEB_SESSION_MAX_AGE", "600")
    monkeypatch.setenv("KIMI_WEB_SESSION_ABSOLUTE_MAX_AGE", "3600")
    token = _token(client)
    now = time.time()
    # Logged in 3400s ago under a 3600s ceiling: renewal may buy 200 more
    # seconds, not the 600 the idle window would otherwise grant.
    _set(token, created_at=now - 3400, expires_at=now + 100)
    ceiling = now - 3400 + 3600

    client.get("/api/auth/me")
    first = _row(token)["expires_at"]
    assert first == pytest.approx(ceiling, abs=1)

    # And it stays there: a session at the ceiling stops writing rows.
    client.get("/api/auth/me")
    assert _row(token)["expires_at"] == first


def test_an_expired_session_is_not_resurrected(client: TestClient) -> None:
    """Renewal must never hand a dead token a new lease."""
    token = _token(client)
    now = time.time()
    _set(token, expires_at=now - 1)

    resp = client.get("/api/auth/me")

    assert resp.status_code == 401
    db = get_db()
    try:
        assert get_user_session_row(db, token) is None  # get_user_session cleans it up
    finally:
        db.close()


def test_the_window_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Desktop installs want a month; a shared server may want a day."""
    monkeypatch.setenv("KIMI_WEB_SESSION_MAX_AGE", "604800")
    assert session_policy.session_max_age() == 604800
    monkeypatch.setenv("KIMI_WEB_SESSION_MAX_AGE", "not-a-number")
    assert session_policy.session_max_age() == session_policy.DEFAULT_MAX_AGE
    monkeypatch.setenv("KIMI_WEB_SESSION_MAX_AGE", "-5")
    assert session_policy.session_max_age() == session_policy.DEFAULT_MAX_AGE


def test_the_ceiling_never_undercuts_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ceiling below the idle window would expire live sessions."""
    monkeypatch.setenv("KIMI_WEB_SESSION_MAX_AGE", "86400")
    monkeypatch.setenv("KIMI_WEB_SESSION_ABSOLUTE_MAX_AGE", "600")
    assert session_policy.session_absolute_max_age() == 86400
