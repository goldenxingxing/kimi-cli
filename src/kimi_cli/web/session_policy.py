"""How long a browser session lives, and when it gets extended.

A session used to be a hard 24-hour window: ``expires_at`` was stamped at
login and nothing ever moved it. A tab left open across the boundary was
logged out mid-use no matter how busy it had been — the symptom people
reported was "come back the next day, refresh, and you're on the login page
again".

The window is still 24 hours of *idleness* by default, but every request
carrying a live cookie slides the deadline forward once less than half of it
is left. Two bounds keep that from becoming a session that never ends:

* ``KIMI_WEB_SESSION_MAX_AGE`` — the idle window (default 24h).
* ``KIMI_WEB_SESSION_ABSOLUTE_MAX_AGE`` — measured from login, never extended
  (default 30 days). A stolen cookie dies at this line even if it is used
  every minute.

Both are read per call rather than at import so a test (or a restart-free
config change) sees the current value.
"""

from __future__ import annotations

import os

#: Name of the cookie carrying the session token.
COOKIE_NAME = "kimi_session"

#: Idle window when nothing is configured (seconds).
DEFAULT_MAX_AGE = 24 * 60 * 60

#: Hard ceiling measured from login when nothing is configured (seconds).
DEFAULT_ABSOLUTE_MAX_AGE = 30 * 24 * 60 * 60

#: Renew once less than this fraction of the idle window is left. At 0.5 a
#: browser that is used at all every twelve hours never sees a login page.
RENEW_RATIO = 0.5

#: Refuse to honour an absurdly short window; five minutes is already brutal.
_MIN_MAX_AGE = 300


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def session_max_age() -> int:
    """The idle window, in seconds."""
    return max(_MIN_MAX_AGE, _int_env("KIMI_WEB_SESSION_MAX_AGE", DEFAULT_MAX_AGE))


def session_absolute_max_age() -> int:
    """The ceiling measured from login, in seconds.

    Never smaller than the idle window: a ceiling below the window it bounds
    would expire sessions the window says are still alive.
    """
    configured = _int_env("KIMI_WEB_SESSION_ABSOLUTE_MAX_AGE", DEFAULT_ABSOLUTE_MAX_AGE)
    return max(configured, session_max_age())


def next_expiry(
    *,
    now: float,
    created_at: float,
    expires_at: float,
    max_age: int,
    absolute_max_age: int,
) -> float | None:
    """Return the new ``expires_at`` for a session, or ``None`` to leave it be.

    ``None`` means one of: the session is already dead, more than
    ``RENEW_RATIO`` of the window is still left, or the absolute ceiling has
    nothing more to give.
    """
    if expires_at <= now:
        return None
    if expires_at - now > max_age * RENEW_RATIO:
        return None
    ceiling = created_at + absolute_max_age
    candidate = min(now + max_age, ceiling)
    # A renewal that buys under a minute is churn: it writes a row and sets a
    # cookie on every single request once the ceiling is close.
    if candidate <= expires_at + 60:
        return None
    return candidate


def build_cookie_header(token: str, max_age: int) -> str:
    """The ``Set-Cookie`` value, matching what the login route writes."""
    return f"{COOKIE_NAME}={token}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=lax"


__all__ = [
    "COOKIE_NAME",
    "DEFAULT_ABSOLUTE_MAX_AGE",
    "DEFAULT_MAX_AGE",
    "RENEW_RATIO",
    "build_cookie_header",
    "next_expiry",
    "session_absolute_max_age",
    "session_max_age",
]
