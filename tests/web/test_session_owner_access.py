"""A session id is not an authorization.

Listing has always filtered by owner. The per-session routes checked nothing,
so in a multi-user deployment any logged-in user who knew another user's
session id could read that user's work-dir files, fork the session, retitle it
or delete it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from kimi_cli.web.api.sessions import ensure_session_access, may_access_session


def _session(owner_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(owner_id=owner_id)


@pytest.fixture
def as_user(monkeypatch: pytest.MonkeyPatch):
    def _set(user: dict[str, str] | None) -> None:
        monkeypatch.setattr("kimi_cli.web.user_auth.user_from_connection", lambda _c: user)

    return _set


def test_another_users_session_is_not_found(as_user) -> None:
    as_user({"id": "user-b", "role": "user"})

    with pytest.raises(HTTPException) as raised:
        ensure_session_access(_session("user-a"), SimpleNamespace())

    # 404, not 403: the answer must not confirm that the id exists.
    assert raised.value.status_code == 404


def test_the_owner_gets_their_own_session(as_user) -> None:
    as_user({"id": "user-a", "role": "user"})

    ensure_session_access(_session("user-a"), SimpleNamespace())


def test_an_admin_may_reach_any_session(as_user) -> None:
    as_user({"id": "root", "role": "admin"})

    ensure_session_access(_session("user-a"), SimpleNamespace())


def test_an_anonymous_caller_cannot_reach_an_owned_session(as_user) -> None:
    as_user(None)

    with pytest.raises(HTTPException):
        ensure_session_access(_session("user-a"), SimpleNamespace())


def test_a_session_with_no_owner_stays_reachable_anonymously(as_user) -> None:
    """Single-user installs, static-token deployments, and old sessions.

    Nothing there records an owner and nobody is logged in; the auth
    middleware is the only gate, and that is unchanged.
    """
    as_user(None)

    ensure_session_access(_session(None), SimpleNamespace())


def test_a_logged_in_user_does_not_get_the_ownerless_exception(as_user) -> None:
    """It has to agree with the listing, which hides these from them anyway.

    list_sessions keeps only sessions whose owner is the caller, so an
    ownerless one never appears in any logged-in user's list. Letting one be
    opened by id widened access without giving anybody back their own history.
    """
    as_user({"id": "user-b", "role": "user"})

    with pytest.raises(HTTPException):
        ensure_session_access(_session(None), SimpleNamespace())


def test_the_socket_and_the_http_routes_ask_the_same_question() -> None:
    """The websocket used to carry its own copy of this rule and skip the
    ownerless case entirely, so a session hidden from a user's list was still
    streamable to them by id."""
    other = {"id": "user-b", "role": "user"}
    owner = {"id": "user-a", "role": "user"}
    admin = {"id": "root", "role": "admin"}

    assert may_access_session(_session("user-a"), owner)
    assert not may_access_session(_session("user-a"), other)
    assert may_access_session(_session("user-a"), admin)
    assert not may_access_session(_session("user-a"), None)

    # The ownerless exception is for callers who are not logged in at all.
    assert may_access_session(_session(None), None)
    assert not may_access_session(_session(None), other)
    assert may_access_session(_session(None), admin)
