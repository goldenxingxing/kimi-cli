"""Resolve the single user-level Wiki namespace."""

from __future__ import annotations

import os
from pathlib import Path

WIKI_SCHEMA_VERSION = 1


def resolve_wiki_root(*, app_data: Path | None = None) -> Path:
    """Return the configured global Wiki root for the current user."""
    configured = os.environ.get("OPENKIMO_WIKI_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    if app_data is None:
        app_data = Path(os.environ["OPENKIMO_APP_DATA_DIR"])
    return (app_data / "users" / "default" / "wiki").resolve()
