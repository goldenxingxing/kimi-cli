"""CRUD operations for user and session management."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
import uuid
from typing import Any

#: Cost of the stdlib KDF. Plain SHA-256 was the fallback before, and a fast
#: general-purpose digest is not a password KDF at any salt: a GPU does
#: billions of those a second. PBKDF2 is in the stdlib and has a work factor,
#: which is the property that matters here.
_PBKDF2_ROUNDS = 600_000

#: Prefixes written by this module rather than by passlib. Always understood,
#: whether or not passlib is importable — see verify_password.
_STDLIB_SCHEMES = ("pbkdf2_sha256$", "sha256$")


def _hash_pbkdf2(plain: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def _verify_stdlib(plain: str, hashed: str) -> bool:
    """Verify a hash this module wrote: PBKDF2, or the older salted SHA-256."""
    try:
        scheme, *rest = hashed.split("$")
        if scheme == "pbkdf2_sha256":
            rounds, salt_hex, digest = rest
            # Bounded, because the number comes out of the stored row. This
            # module only ever writes _PBKDF2_ROUNDS, so anything wildly larger
            # is a corrupt or tampered hash — and deriving with it would sit in
            # pbkdf2_hmac for minutes on the login path, wedging every login
            # behind it. A count far below is not a password hash either.
            iterations = int(rounds)
            if not 1000 <= iterations <= _PBKDF2_ROUNDS * 4:
                return False
            computed = hashlib.pbkdf2_hmac(
                "sha256", plain.encode(), bytes.fromhex(salt_hex), iterations
            )
            return hmac.compare_digest(computed.hex(), digest)
        if scheme == "sha256":
            # Accounts created by the first fallback still have to log in;
            # their hashes are upgraded the next time the password is changed,
            # not silently invalidated here.
            salt, digest = rest
            expected = hashlib.sha256(f"{salt}:{plain}".encode()).hexdigest()
            return hmac.compare_digest(expected, digest)
    except Exception:
        return False
    return False


# Password hashing: prefer passlib/bcrypt, fall back to the stdlib KDF above.
try:
    from passlib.context import CryptContext as _CryptContext  # type: ignore[import-untyped]

    _pwd_context = _CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception as _exc:  # noqa: BLE001 - see below
    # Broad on purpose: a bcrypt backend that fails to *initialise* lands here
    # as well as a missing passlib, and taking the whole server down over it
    # would be worse. But it is a downgrade, so it is said out loud rather than
    # happening in silence — that was the actual problem: every password
    # created afterwards was hashed differently and nothing anywhere said so.
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "passlib/bcrypt unavailable (%s); falling back to PBKDF2 password hashing",
        _exc,
    )
    _pwd_context = None


_warned_unreadable_hash = False


def _warn_unreadable_hash_once() -> None:
    global _warned_unreadable_hash
    if _warned_unreadable_hash:
        return
    _warned_unreadable_hash = True
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "A stored password hash needs passlib, which is not importable here; "
        "logins for accounts created with it will fail until it is installed."
    )


def hash_password(plain: str) -> str:
    """Hash a plaintext password."""
    if _pwd_context is not None:
        return _pwd_context.hash(plain)  # type: ignore[no-any-return]
    return _hash_pbkdf2(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against its stored hash.

    The scheme is decided by the hash, not by what happens to be installed.
    passlib is not a declared dependency, so every account created on a host
    without it carries a `pbkdf2_sha256$...` hash — and the day passlib became
    importable (a transitive bump, a rebuilt image), a bcrypt-only CryptContext
    would raise UnknownHashError on all of them. That is a 500, not a 401, for
    every user including the admin, with nothing to fall back on.
    """
    if hashed.startswith(_STDLIB_SCHEMES):
        return _verify_stdlib(plain, hashed)
    if _pwd_context is None:
        # A bcrypt hash and no passlib to read it: nobody with a password set
        # on a host that had passlib can log in, admin included. Nothing can be
        # done about that here — but it must not look like a wrong password, or
        # the outage is undiagnosable from the logs.
        _warn_unreadable_hash_once()
        return False
    try:
        return _pwd_context.verify(plain, hashed)  # type: ignore[no-any-return]
    except Exception:
        return False


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


def get_user_by_username(db: sqlite3.Connection, username: str) -> dict[str, Any] | None:
    """Return user dict for *username*, or ``None`` if not found."""
    row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return _row_to_dict(row)


def get_user_by_id(db: sqlite3.Connection, user_id: str) -> dict[str, Any] | None:
    """Return user dict for *user_id*, or ``None`` if not found."""
    row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_dict(row)


def create_user(
    db: sqlite3.Connection,
    username: str,
    password: str,
    role: str = "user",
) -> dict[str, Any]:
    """Create a new user and return the resulting user dict."""
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    now = time.time()
    db.execute(
        """
        INSERT INTO users (id, username, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, username, password_hash, role, 1, now),
    )
    db.commit()
    user = get_user_by_id(db, user_id)
    assert user is not None
    return user


def update_user(
    db: sqlite3.Connection,
    user_id: str,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Update user fields.

    Accepted keyword arguments: ``password``, ``role``, ``is_active``.
    Returns the updated user dict, or ``None`` if the user does not exist.

    Changing the password also logs the user out everywhere.
    """
    if not kwargs:
        return get_user_by_id(db, user_id)

    set_clauses: list[str] = []
    params: list[Any] = []

    if "password" in kwargs:
        set_clauses.append("password_hash = ?")
        params.append(hash_password(kwargs["password"]))
    if "role" in kwargs:
        set_clauses.append("role = ?")
        params.append(kwargs["role"])
    if "is_active" in kwargs:
        set_clauses.append("is_active = ?")
        params.append(1 if kwargs["is_active"] else 0)

    if not set_clauses:
        return get_user_by_id(db, user_id)

    params.append(user_id)
    db.execute(
        f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?",  # noqa: S608
        params,
    )
    if "password" in kwargs:
        # A password change has to end the sessions opened with the old one.
        # Deactivating an account already does (get_user_session re-checks
        # is_active); a reset did not, so a stolen cookie outlived the reset
        # that was meant to revoke it — and with sliding renewal, for as long
        # as it kept being used, up to the absolute ceiling.
        db.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
    db.commit()
    return get_user_by_id(db, user_id)


def delete_user(db: sqlite3.Connection, user_id: str) -> bool:
    """Delete a user and all their sessions.  Returns ``True`` if deleted."""
    # Remove sessions first to avoid orphaned rows
    db.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
    cursor = db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    return cursor.rowcount > 0


def list_users(db: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all users ordered by creation time."""
    rows = db.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


def create_user_session(
    db: sqlite3.Connection,
    user_id: str,
    expires_in_seconds: int = 86400,
) -> str:
    """Create a new user session token and persist it.  Returns the token."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    expires_at = now + expires_in_seconds
    db.execute(
        """
        INSERT INTO user_sessions (token, user_id, created_at, expires_at)
        VALUES (?, ?, ?, ?)
        """,
        (token, user_id, now, expires_at),
    )
    db.commit()
    return token


def get_user_session(db: sqlite3.Connection, token: str) -> dict[str, Any] | None:
    """Return the user dict associated with *token*, or ``None`` if expired/missing."""
    row = db.execute("SELECT * FROM user_sessions WHERE token = ?", (token,)).fetchone()
    if row is None:
        return None

    session = dict(row)
    if session["expires_at"] < time.time():
        # Session expired — clean up lazily
        db.execute("DELETE FROM user_sessions WHERE token = ?", (token,))
        db.commit()
        return None

    user = get_user_by_id(db, session["user_id"])
    if user is None or not user.get("is_active"):
        return None
    return user


def get_user_session_row(db: sqlite3.Connection, token: str) -> dict[str, Any] | None:
    """Return the raw ``user_sessions`` row for *token*, expired or not.

    Separate from :func:`get_user_session` because renewal needs the timestamps
    the user dict does not carry, and must be able to tell "expired" from
    "never existed" without deleting anything.
    """
    row = db.execute("SELECT * FROM user_sessions WHERE token = ?", (token,)).fetchone()
    return dict(row) if row is not None else None


def set_user_session_expiry(db: sqlite3.Connection, token: str, expires_at: float) -> None:
    """Move a session's deadline. Silently does nothing if the row is gone."""
    db.execute(
        "UPDATE user_sessions SET expires_at = ? WHERE token = ?",
        (expires_at, token),
    )
    db.commit()


def delete_user_session(db: sqlite3.Connection, token: str) -> None:
    """Delete a user session by token."""
    db.execute("DELETE FROM user_sessions WHERE token = ?", (token,))
    db.commit()


# ---------------------------------------------------------------------------
# Branding CRUD
# ---------------------------------------------------------------------------

# All valid branding keys
BRANDING_KEYS = {"brand_name", "version", "page_title", "logo_url", "logo", "favicon"}


def get_branding(db: sqlite3.Connection) -> dict[str, str | None]:
    """Return all branding settings. Unset keys map to ``None``."""
    rows = db.execute("SELECT key, value FROM branding").fetchall()
    result: dict[str, str | None] = {k: None for k in BRANDING_KEYS}
    for row in rows:
        k = row["key"]
        if k in BRANDING_KEYS:
            result[k] = row["value"]
    return result


def upsert_branding(db: sqlite3.Connection, settings: dict[str, str | None]) -> None:
    """Batch-update branding settings. A ``None`` value deletes the key."""
    for key, value in settings.items():
        if key not in BRANDING_KEYS:
            continue
        if value is None or value == "":
            db.execute("DELETE FROM branding WHERE key = ?", (key,))
        else:
            db.execute(
                "INSERT INTO branding (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
    db.commit()


def delete_all_branding(db: sqlite3.Connection) -> None:
    """Delete all branding settings, restoring defaults."""
    db.execute("DELETE FROM branding")
    db.commit()


__all__ = [
    "BRANDING_KEYS",
    "create_user",
    "create_user_session",
    "delete_all_branding",
    "delete_user",
    "delete_user_session",
    "get_branding",
    "get_user_by_id",
    "get_user_by_username",
    "get_user_session",
    "hash_password",
    "list_users",
    "update_user",
    "upsert_branding",
    "verify_password",
]
