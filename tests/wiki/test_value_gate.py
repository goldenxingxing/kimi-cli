from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest

from kimi_cli.wiki.models import PageChange, SourceRef, WikiCandidate, WikiPage

_SESSION_ID = UUID("123e4567-e89b-12d3-a456-426614174000")
_SOURCE_HASH = "sha256:" + "a" * 64
_NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)


def _source() -> SourceRef:
    return SourceRef(kind="conversation", session_id=_SESSION_ID, content_hash=_SOURCE_HASH)


def _candidate(
    *,
    path: str = "concepts/atomic-writes.md",
    body: str = "Atomic replacement protects readers from incomplete files.\n",
    value: Literal["high", "medium", "low"] = "high",
    revision: int = 1,
    expected_revision: int | None = None,
) -> WikiCandidate:
    source = _source()
    page = WikiPage(
        logical_path=path,
        title="Atomic writes",
        created=_NOW,
        updated=_NOW,
        tags=["wiki"],
        sources=[source],
        revision=revision,
        body=body,
    )
    return WikiCandidate(
        summary="Record durable atomic write guidance",
        pages=[PageChange(page=page, expected_revision=expected_revision)],
        sources=[source],
        value=value,
    )


@pytest.fixture
def manager(tmp_path: Path):
    from kimi_cli.wiki.manager import WikiManager

    instance = WikiManager(tmp_path / "wiki", wal=False)
    yield instance
    instance.close()


def _context(**updates: object):
    from kimi_cli.wiki.value_gate import WikiContext

    values: dict[str, object] = {
        "session_id": _SESSION_ID,
        "cross_turn_utility": True,
        "stable": True,
        "user_confirmed": True,
    }
    values.update(updates)
    return WikiContext.model_validate(values)


@pytest.mark.parametrize("value", ["medium", "low"])
def test_gate_discards_non_high_value_without_queue(
    manager, value: Literal["medium", "low"]
) -> None:
    from kimi_cli.wiki.value_gate import DiscardedCandidate

    result = manager.prepare(_candidate(value=value), _context())

    assert isinstance(result, DiscardedCandidate)
    assert result.reason == "low_value"
    assert not (manager.layout.metadata / "pending").exists()


def test_gate_discards_one_turn_only_and_unstable_candidates(manager) -> None:
    assert manager.prepare(_candidate(), _context(cross_turn_utility=False)).reason == "low_value"
    assert manager.prepare(_candidate(), _context(stable=False)).reason == "unstable"


def test_gate_requires_explicit_grounding_for_conversation_and_web_sources(manager) -> None:
    assert manager.prepare(_candidate(), _context(user_confirmed=False)).reason == "ungrounded"


def test_gate_discards_secret_bearing_page_in_memory(manager) -> None:
    candidate = _candidate(body="API key: sk-abcdefghijklmnopqrstuvwxyz\n")

    result = manager.prepare(candidate, _context())

    assert result.reason == "sensitive"
    assert not (manager.layout.metadata / "pending").exists()


def test_gate_discards_machine_specific_summary_before_audit(manager) -> None:
    candidate = _candidate().model_copy(
        update={"summary": "Remember /Users/example/private/wiki.md"}
    )

    assert manager.prepare(candidate, _context()).reason == "sensitive"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "API key: sk-abcdefghijklmnopqrstuvwxyz"),
        ("title", "Private notes at /Users/example/wiki.md"),
        ("tags", ["safe", "refresh_token=do-not-store"]),
        ("tags", ["safe", r"C:\Users\example\wiki.md"]),
    ],
)
def test_gate_scans_all_string_frontmatter_for_secrets_and_machine_paths(
    manager, field: str, value: object
) -> None:
    candidate = _candidate()
    page = candidate.pages[0].page.model_copy(update={field: value})
    candidate = candidate.model_copy(
        update={"pages": [PageChange(page=page, expected_revision=None)]}
    )

    assert manager.prepare(candidate, _context()).reason == "sensitive"


def test_gate_discards_duplicate_without_creating_review_state(manager) -> None:
    from kimi_cli.wiki.value_gate import DiscardedCandidate

    first = manager.prepare(_candidate(), _context())
    manager.commit(first)
    duplicate = _candidate(revision=2, expected_revision=1)

    result = manager.prepare(duplicate, _context())

    assert isinstance(result, DiscardedCandidate)
    assert result.reason == "duplicate"
    assert not (manager.layout.metadata / "pending").exists()


def test_gate_discards_same_content_at_a_second_path_even_with_new_source(manager) -> None:
    manager.commit(manager.prepare(_candidate(), _context()))
    duplicate = _candidate(
        path="queries/atomic-writes.md",
        body="Atomic replacement protects readers from incomplete files.\n",
    )
    second_source = _source().model_copy(update={"content_hash": "sha256:" + "b" * 64})
    page = duplicate.pages[0].page.model_copy(
        update={"title": "Atomic replacement note", "sources": [second_source]}
    )
    duplicate = duplicate.model_copy(
        update={
            "pages": [PageChange(page=page, expected_revision=None)],
            "sources": [second_source],
        }
    )

    assert manager.prepare(duplicate, _context()).reason == "duplicate"


def test_gate_requires_candidate_sources_to_cover_each_page(manager) -> None:
    candidate = _candidate()
    uncovered = candidate.model_copy(update={"sources": []})

    assert manager.prepare(uncovered, _context()).reason == "ungrounded"


def test_gate_rejects_unresolvable_workspace_provenance(manager) -> None:
    workspace_source = SourceRef(
        kind="workspace-file",
        workspace_id=UUID("423e4567-e89b-12d3-a456-426614174000"),
        path="docs/source.md",
        content_hash=_SOURCE_HASH,
    )
    candidate = _candidate()
    page = candidate.pages[0].page.model_copy(update={"sources": [workspace_source]})
    candidate = candidate.model_copy(
        update={
            "pages": [PageChange(page=page, expected_revision=None)],
            "sources": [workspace_source],
        }
    )

    assert manager.prepare(candidate, _context()).reason == "ungrounded"


def test_gate_revalidates_source_frontmatter_before_workspace_resolution(manager) -> None:
    workspace_source = SourceRef(
        kind="workspace-file",
        workspace_id=UUID("423e4567-e89b-12d3-a456-426614174000"),
        path="docs/source.md",
        content_hash=_SOURCE_HASH,
    ).model_copy(update={"path": "/Users/example/private.md"})
    candidate = _candidate()
    page = candidate.pages[0].page.model_copy(update={"sources": [workspace_source]})
    candidate = candidate.model_copy(
        update={
            "pages": [PageChange(page=page, expected_revision=None)],
            "sources": [workspace_source],
        }
    )

    assert manager.prepare(candidate, _context()).reason == "sensitive"


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate().model_copy(update={"summary": "x" * 501}),
        _candidate().model_copy(update={"pages": []}),
    ],
)
def test_gate_revalidates_complete_candidate_shape(manager, candidate: WikiCandidate) -> None:
    assert manager.prepare(candidate, _context()).reason == "sensitive"
