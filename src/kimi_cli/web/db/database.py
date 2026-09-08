"""SQLite database connection and initialization for user management."""

from __future__ import annotations

import contextlib
import os
import sqlite3
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

# Database path — reuse get_share_dir() so KIMI_SHARE_DIR env var is respected.
# In container deployments KIMI_SHARE_DIR should point to a mounted volume,
# which keeps both session data and the user database persistent across restarts.
from kimi_cli.share import get_share_dir as _get_share_dir

# The one implementation, in crud, rather than a second copy of the
# passlib-or-fallback dance. Two copies meant the fallback could be fixed in
# one of them and left weak in the other — and this is the copy that hashes the
# very first admin password.
from kimi_cli.web.db.crud import hash_password as _hash_password


def _get_db_path() -> Path:
    share_dir = _get_share_dir()
    return share_dir / "users.db"


_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
"""

_CREATE_USER_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS user_sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""

_CREATE_BRANDING_TABLE = """
CREATE TABLE IF NOT EXISTS branding (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def get_db() -> sqlite3.Connection:
    """Return a SQLite connection to the users database.

    The caller must close it. ``with get_db() as conn:`` does *not* do that —
    sqlite3's context manager wraps the transaction and leaves the connection
    open — which is why :func:`db_session` exists. Row factory is set to
    ``sqlite3.Row`` so rows behave like dicts.
    """
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextlib.contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    """A connection that is committed or rolled back, and then closed.

    ``with get_db() as conn:`` reads as if it did this, and does not: sqlite3's
    own context manager only wraps the transaction. Every request that used it
    left a connection open — a login, an admin call, a branding read — and a
    long-running server ran itself out of descriptors.
    """
    conn = get_db()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the database schema and create the default admin account.

    Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS``.  The
    default admin is only created when the ``users`` table is empty.
    """
    with db_session() as conn:
        conn.execute(_CREATE_USERS_TABLE)
        conn.execute(_CREATE_USER_SESSIONS_TABLE)
        conn.execute(_CREATE_BRANDING_TABLE)
        conn.commit()

        # Create default admin only when table is empty
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        if row[0] == 0:
            admin_id = str(uuid.uuid4())
            # The documented default stays the default — README, docs and
            # deploy.sh all name it — but a deployment that does not want a
            # published password on a privileged account can now say so
            # without patching the source.
            initial_password = os.environ.get("KIMI_INITIAL_ADMIN_PASSWORD") or "admin123"
            password_hash = _hash_password(initial_password)
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (admin_id, "admin", password_hash, "admin", 1, time.time()),
            )
            conn.commit()


__all__ = [
    "db_session",
    "get_db",
    "init_db",
]
