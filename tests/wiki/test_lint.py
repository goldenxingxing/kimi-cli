from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from kimi_cli.wiki.models import SourceRef, WikiPage
from kimi_cli.wiki.schema import render_page

_NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)
_SESSION_ID = UUID("323e4567-e89b-12d3-a456-426614174000")


def _page(
    logical_path: str,
    body: str,
    *,
    sources: bool = True,
) -> WikiPage:
    source_list = (
        [
            SourceRef(
                kind="conversation",
                session_id=_SESSION_ID,
                content_hash="sha256:" + "d" * 64,
            )
        ]
        if sources
        else []
    )
    return WikiPage(
        logical_path=logical_path,
        title=logical_path.rsplit("/", 1)[1].removesuffix(".md"),
        created=_NOW,
        updated=_NOW,
        tags=["lint"],
        sources=source_list,
        revision=1,
        body=body,
    )


@pytest.fixture
def manager(tmp_path: Path):
    from kimi_cli.wiki.manager import WikiManager

    instance = WikiManager(tmp_path / "wiki", wal=False)
    yield instance
    instance.close()


def _write(manager, page: WikiPage) -> None:
    target = manager.layout.root / page.logical_path
    target.write_text(render_page(page), encoding="utf-8")


def test_lint_reports_dead_links_orphans_duplicates_conflicts_and_provenance(manager) -> None:
    repeated = "A durable claim repeated across pages for lint detection."
    _write(
        manager,
        _page(
            "concepts/alpha.md",
            f"{repeated}\n\nSee [[entities/missing]].\n",
        ),
    )
    _write(manager, _page("concepts/beta.md", f"{repeated}\n"))
    _write(
        manager,
        _page(
            "entities/conflicted.md",
            "Both positions remain.\n\n## Conflict\n\nPosition A and position B.\n",
            sources=False,
        ),
    )

    report = manager.lint(None)
    codes = {issue.code for issue in report.issues}

    assert {
        "dead_link",
        "orphan",
        "duplicate_claim",
        "conflict_marker",
        "missing_provenance",
    } <= codes
    assert report.scanned_pages == 3
    assert report.modified_pages == 0


def test_lint_reports_malformed_pages_without_modifying_them(manager) -> None:
    target = manager.layout.root / "concepts" / "broken.md"
    original = b"not frontmatter\n"
    target.write_bytes(original)

    report = manager.lint("concepts")

    assert any(
        issue.code == "malformed_page" and issue.logical_path == "concepts/broken.md"
        for issue in report.issues
    )
    assert target.read_bytes() == original
    assert report.modified_pages == 0


def test_link_to_malformed_page_is_reported_as_dead(manager) -> None:
    _write(manager, _page("concepts/source.md", "See [[concepts/broken]].\n"))
    broken = manager.layout.root / "concepts" / "broken.md"
    broken.write_text("not valid Wiki Markdown\n", encoding="utf-8")

    report = manager.lint(None)

    assert any(
        issue.code == "dead_link"
        and issue.logical_path == "concepts/source.md"
        and issue.related_paths == ("concepts/broken.md",)
        for issue in report.issues
    )


def test_lint_reports_non_utf8_page_as_malformed(manager) -> None:
    target = manager.layout.root / "concepts" / "broken.md"
    original = (
        b"---\ntitle: Broken\ncreated: 2026-07-24T12:00:00+00:00\n"
        b"updated: 2026-07-24T12:00:00+00:00\ntags: []\nsources: []\n"
        b"revision: 1\n---\nInvalid: \xff\n"
    )
    target.write_bytes(original)

    report = manager.lint(None)

    assert any(issue.code == "malformed_page" for issue in report.issues)
    assert target.read_bytes() == original


def test_lint_scope_is_validated_and_limits_scan(manager) -> None:
    _write(manager, _page("concepts/alpha.md", "Concept alpha.\n"))
    _write(manager, _page("entities/beta.md", "Entity beta.\n"))

    report = manager.lint("concepts")

    assert report.scanned_pages == 1
    assert all(issue.logical_path.startswith("concepts/") for issue in report.issues)
    with pytest.raises(ValueError, match="scope"):
        manager.lint("../outside")
