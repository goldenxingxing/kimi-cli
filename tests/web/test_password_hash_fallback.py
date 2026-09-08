"""The fallback used when passlib is missing is still a password hash.

passlib is not a declared dependency of this project, so on most hosts the
"fallback" *is* the production path.

It used to be salted SHA-256 — a fast general-purpose digest, which a GPU does
billions of per second — chosen silently by a broad `except` that also catches
a bcrypt backend failing to initialise. Nothing in the logs said the downgrade
had happened, so a deployment could be storing cheap-to-crack hashes without
anyone knowing.
"""

from __future__ import annotations

import hashlib
import importlib
import sys

import pytest

from kimi_cli.web.db import crud


@pytest.fixture
def fallback_crud(monkeypatch: pytest.MonkeyPatch):
    """`crud` as it is on a host without passlib."""
    monkeypatch.setitem(sys.modules, "passlib", None)
    monkeypatch.setitem(sys.modules, "passlib.context", None)
    module = importlib.reload(crud)
    try:
        yield module
    finally:
        monkeypatch.undo()
        importlib.reload(crud)


def test_the_fallback_is_a_kdf_with_a_work_factor(fallback_crud) -> None:
    hashed = fallback_crud.hash_password("hunter2")

    assert hashed.startswith("pbkdf2_sha256$")
    assert fallback_crud.verify_password("hunter2", hashed)
    assert not fallback_crud.verify_password("hunter3", hashed)


def test_passwords_hashed_by_the_old_fallback_still_log_in(fallback_crud) -> None:
    """Upgrading the scheme must not lock out the accounts it was used for."""
    salt = "ab" * 8
    digest = hashlib.sha256(f"{salt}:hunter2".encode()).hexdigest()
    legacy = f"sha256${salt}${digest}"

    assert fallback_crud.verify_password("hunter2", legacy)
    assert not fallback_crud.verify_password("hunter3", legacy)


def test_a_hash_it_cannot_parse_is_a_failed_verification(fallback_crud) -> None:
    assert not fallback_crud.verify_password("hunter2", "not-a-hash")
    assert not fallback_crud.verify_password("hunter2", "")


def test_a_stdlib_hash_still_verifies_once_passlib_arrives(monkeypatch) -> None:
    """The day passlib becomes importable must not lock everyone out.

    Every account created without it carries a `pbkdf2_sha256$...` hash, and a
    bcrypt-only CryptContext raises UnknownHashError on those — a 500 for every
    user, admin included, with no way back in. The scheme is chosen by the
    hash, not by what happens to be installed.
    """

    class _BcryptOnlyContext:
        def hash(self, _plain: str) -> str:  # pragma: no cover - unused here
            return "$2b$12$notarealbcrypthash"

        def verify(self, _plain: str, _hashed: str) -> bool:
            raise ValueError("hash could not be identified")

    stdlib_hash = crud._hash_pbkdf2("hunter2")
    monkeypatch.setattr(crud, "_pwd_context", _BcryptOnlyContext())

    assert crud.verify_password("hunter2", stdlib_hash)
    assert not crud.verify_password("hunter3", stdlib_hash)


def test_a_hash_passlib_cannot_read_is_a_failed_login_not_a_500(monkeypatch) -> None:
    class _AngryContext:
        def verify(self, _plain: str, _hashed: str) -> bool:
            raise ValueError("hash could not be identified")

    monkeypatch.setattr(crud, "_pwd_context", _AngryContext())

    assert not crud.verify_password("hunter2", "$2b$12$something-unreadable")
