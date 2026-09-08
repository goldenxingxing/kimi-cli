"""User authentication API endpoints (login / logout / me)."""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.requests import Request
from pydantic import BaseModel

from kimi_cli.web.db.crud import (
    create_user_session,
    delete_user_session,
    get_user_by_username,
    hash_password,
    verify_password,
)
from kimi_cli.web.db.database import db_session
from kimi_cli.web.session_policy import COOKIE_NAME as _COOKIE_NAME
from kimi_cli.web.session_policy import session_max_age
from kimi_cli.web.user_auth import require_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    role: str


@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    """A hash of nothing, to spend the same time on a username that is not there.

    Computed on first use rather than at import so a server start does not pay
    for a bcrypt round it may never need.
    """
    return hash_password(secrets.token_urlsafe(32))


@router.post("/login", summary="Log in and obtain a session cookie")
def login(body: LoginRequest, response: Response) -> UserResponse:
    """Authenticate with username and password.

    On success, sets an ``HttpOnly`` ``kimi_session`` cookie and returns basic
    user information.  Returns 401 on invalid credentials or inactive account.

    A plain ``def``, not ``async def``: everything in here is blocking, and the
    expensive part is deliberately so. Verifying a password is a KDF — bcrypt
    when passlib is present, PBKDF2 at 600k rounds when it is not, which is a
    tenth of a second on a laptop and closer to half on a container vCPU — and
    the equal-time path below now pays it for an unknown username too. On the
    event loop that is a login endpoint anyone can use to stall every socket,
    stream and request in the process. FastAPI runs a sync handler in its
    threadpool, which is where this belongs.
    """
    with db_session() as db:
        user = get_user_by_username(db, body.username)
        if user is None:
            # Pay for a hash anyway. Returning without one answers an unknown
            # username in microseconds while a known one costs a full bcrypt
            # round — the two responses are identical, but the clock says which
            # accounts exist.
            verify_password(body.password, _dummy_password_hash())
        if user is None or not verify_password(body.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )
        if not user.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is disabled",
            )
        max_age = session_max_age()
        token = create_user_session(db, user["id"], expires_in_seconds=max_age)

    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=max_age,
    )
    return UserResponse(
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
    )


@router.post("/logout", summary="Log out and clear the session cookie")
def logout(request: Request, response: Response) -> dict[str, str]:
    """Invalidate the current session and clear the session cookie."""
    token = request.cookies.get(_COOKIE_NAME)
    if token:
        try:
            with db_session() as db:
                delete_user_session(db, token)
        except Exception:
            pass

    response.delete_cookie(key=_COOKIE_NAME, path="/")
    return {"detail": "Logged out"}


@router.get("/me", summary="Get current authenticated user")
async def me(
    user: dict[str, Any] = Depends(require_current_user),
) -> UserResponse:
    """Return the currently authenticated user.  Returns 401 if not logged in."""
    return UserResponse(
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
    )


__all__ = ["router"]
