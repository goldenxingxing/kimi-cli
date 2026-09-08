"""Changing a password has to end the sessions opened with the old one.

Deactivating an account already did — `get_user_session` re-checks `is_active`
on every request — but a reset left every `user_sessions` row in place. With
sliding renewal, a stolen cookie that keeps being used outlives the reset that
was meant to revoke it, up to the 30-day absolute ceiling.
"""

from __future__ import annotations

import sqlite3

import pytest

from kimi_cli.web.db import crud
from kimi_cli.web.db.database import _CREATE_USER_SESSIONS_TABLE, _CREATE_USERS_TABLE


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_USERS_TABLE)
    conn.execute(_CREATE_USER_SESSIONS_TABLE)
    yield conn
    conn.close()


def test_a_password_change_revokes_the_open_sessions(db) -> None:
    user = crud.create_user(db, "someone", "old-password")
    token = crud.create_user_session(db, user["id"], expires_in_seconds=3600)
    assert crud.get_user_session(db, token) is not None

    crud.update_user(db, user["id"], password="new-password")

    assert crud.get_user_session(db, token) is None


def test_another_field_leaves_the_session_alone(db) -> None:
    """Only the password invalidates; renaming a role should not log anyone out."""
    user = crud.create_user(db, "someone", "old-password")
    token = crud.create_user_session(db, user["id"], expires_in_seconds=3600)

    crud.update_user(db, user["id"], role="admin")

    assert crud.get_user_session(db, token) is not None
