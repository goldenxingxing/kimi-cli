"""The SPA catch-all must not serve anything outside the static tree.

`/{full_path:path}` receives the decoded request path, so `%2e%2e` arrives as
`..` and `STATIC_DIR / full_path` walks straight out of the static directory.
These routes sit in front of the API auth middleware — it returns early for
anything that is not `/api/` — so an escape here is an unauthenticated read of
any file the server process can open. Starlette's own StaticFiles carries the
same containment check for the same reason.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from kimi_cli.web.app import STATIC_DIR, create_app

_needs_static = pytest.mark.skipif(
    not (STATIC_DIR / "index.html").exists(),
    reason="web static assets not built",
)


@_needs_static
def test_encoded_dot_segments_cannot_escape_the_static_root() -> None:
    # app.py sits one directory above static/ — a real file, so a vulnerable
    # build answers 200 with this module's own source.
    with TestClient(create_app()) as client:
        response = client.get("/%2e%2e/app.py")

    assert response.status_code == 404
    assert "create_app" not in response.text


@_needs_static
def test_a_real_static_file_is_still_served() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/index.html")

    assert response.status_code == 200
