"""System capabilities API.

Exposes platform-level facts the frontend needs to know about, e.g. whether
Git Bash is available on Windows (required by kimi-cli's Shell tool).
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from kimi_cli.utils.environment import GitBashNotFoundError, _find_git_bash_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GIT_BASH_INSTALL_URL = "https://git-scm.com/downloads/win"

# A "missing" answer can flip to "present" once the user installs Git for
# Windows, so cache negatives only briefly. A "present" answer is sticky: the
# bash.exe location never disappears mid-process under normal conditions.
_NEGATIVE_CACHE_TTL_SECONDS = 30.0

# Process-wide cache. ``None`` means "not yet probed".
# Tuple shape: (git_bash_available, expires_at_monotonic).
_cache_lock = asyncio.Lock()
_cached_result: tuple[bool, float] | None = None


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class SystemCapabilities(BaseModel):
    """Response model for ``GET /api/system/capabilities``."""

    platform: Literal["win32", "darwin", "linux"] | str
    git_bash: bool
    git_bash_install_url: str = GIT_BASH_INSTALL_URL


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _normalized_platform() -> str:
    """Return the normalized ``sys.platform`` value we expose to the frontend.

    On Windows ``sys.platform`` is ``win32``; macOS is ``darwin``; most Linux
    builds are ``linux``. Other values (e.g. ``freebsd``) pass through.
    """
    return sys.platform


async def _detect_git_bash() -> bool:
    """Detect whether git-bash is available on the current process.

    Returns:
        True on non-Windows (irrelevant — shell tool uses /bin/bash or /bin/sh).
        True on Windows when ``_find_git_bash_path()`` resolves.
        False on Windows when the resolver raises ``GitBashNotFoundError``.
    """
    if _normalized_platform() != "win32":
        return True
    try:
        await _find_git_bash_path()
        return True
    except GitBashNotFoundError:
        return False


async def _get_git_bash_cached() -> bool:
    """Return the cached git-bash availability, refreshing as needed."""
    global _cached_result
    async with _cache_lock:
        now = time.monotonic()
        if _cached_result is not None:
            value, expires_at = _cached_result
            if value or now < expires_at:
                # Positive results are sticky (no expiry check); negative
                # results expire after the TTL.
                return value

        value = await _detect_git_bash()
        # Positive answers stick around forever (large ``expires_at``);
        # negatives are revalidated after the TTL window.
        expires_at = float("inf") if value else now + _NEGATIVE_CACHE_TTL_SECONDS
        _cached_result = (value, expires_at)
        return value


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

capabilities_router = APIRouter(prefix="/api/system", tags=["system"])


@capabilities_router.get(
    "/capabilities",
    summary="Return platform-level capabilities for the frontend",
)
async def get_system_capabilities() -> SystemCapabilities:
    """Report platform-level capabilities for UI banners/feature toggles."""
    git_bash = await _get_git_bash_cached()
    return SystemCapabilities(
        platform=_normalized_platform(),
        git_bash=git_bash,
        git_bash_install_url=GIT_BASH_INSTALL_URL,
    )


# Alias matching the spec's preferred export name.
router = capabilities_router

__all__ = ["capabilities_router", "router", "SystemCapabilities"]
