"""Deterministic admission policy for durable global Wiki knowledge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from kimi_cli.wiki.models import SourceRef, UnsafeWikiPage, WikiCandidate, WikiPage
from kimi_cli.wiki.schema import render_page, validate_logical_page

DiscardReason = Literal["low_value", "unstable", "ungrounded", "sensitive", "duplicate"]
Operation = Literal["remember", "ingest"]
_SECRET_TEXT = re.compile(
    r"(?i)(?:"
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----|"
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b|"
    r"(?<![\w?&-])(?:access[_-]?token|api[_-]?key|auth[_-]?token|authorization|"
    r"client[_-]?secret|cookie|id[_-]?token|password|private[_-]?key|"
    r"refresh[_-]?token|secret|secret[_-]?key|session[_-]?token|"
    r"user[_-]?password|user[_-]?token)\s*[:=]\s*\S+"
    r")"
)


class WikiContext(BaseModel):
    """Structured, current-session evidence used by the value gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: UUID
    cross_turn_utility: bool
    stable: bool
    user_confirmed: bool = False
    reliable_source: bool = False
    operation: Operation = "remember"
    conflicting_pages: tuple[str, ...] = ()

    @field_validator("conflicting_pages")
    @classmethod
    def validate_conflicting_pages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(validate_logical_page(path).as_posix() for path in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("conflicting pages must be unique")
        return normalized


@dataclass(frozen=True, slots=True)
class DiscardedCandidate:
    """A rejected candidate that exists only in caller memory."""

    reason: DiscardReason
    summary: str


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Internal result of deterministic value and safety evaluation."""

    accepted: bool
    reason: DiscardReason | None = None


def contains_sensitive_text(text: str) -> bool:
    """Return whether unstructured input visibly carries credential material."""
    return bool(_SECRET_TEXT.search(text))


def evaluate_candidate(
    candidate: WikiCandidate,
    context: WikiContext,
    existing_pages: tuple[WikiPage, ...],
) -> GateDecision:
    """Apply the high-value, grounded, safe, and novel admission policy."""
    if candidate.value != "high" or not context.cross_turn_utility:
        return GateDecision(False, "low_value")
    if not context.stable:
        return GateDecision(False, "unstable")
    if not _is_grounded(candidate, context):
        return GateDecision(False, "ungrounded")
    if not _summary_is_safe(candidate) or not _pages_are_safe(candidate):
        return GateDecision(False, "sensitive")
    if len(duplicate_candidate_paths(candidate, existing_pages)) == len(candidate.pages):
        return GateDecision(False, "duplicate")
    return GateDecision(True)


def duplicate_candidate_paths(
    candidate: WikiCandidate,
    existing_pages: tuple[WikiPage, ...],
) -> tuple[str, ...]:
    """Return proposal pages that add no semantic content to the snapshot."""
    existing_by_path = {page.logical_path: page for page in existing_pages}
    existing_content = {_knowledge_fingerprint(page) for page in existing_pages}
    duplicates: list[str] = []
    for change in candidate.pages:
        page = change.page
        current = existing_by_path.get(page.logical_path)
        fingerprint = _content_fingerprint(page)
        if current is not None and _content_fingerprint(current) == fingerprint:
            current_sources = {_source_key(source) for source in current.sources}
            proposed_sources = {_source_key(source) for source in page.sources}
            if proposed_sources.issubset(current_sources):
                duplicates.append(page.logical_path)
        elif _knowledge_fingerprint(page) in existing_content:
            duplicates.append(page.logical_path)
    return tuple(duplicates)


def _is_grounded(candidate: WikiCandidate, context: WikiContext) -> bool:
    if not candidate.sources or any(not change.page.sources for change in candidate.pages):
        return False
    candidate_sources = {_source_key(source) for source in candidate.sources}
    page_sources = {
        _source_key(source) for change in candidate.pages for source in change.page.sources
    }
    if not page_sources.issubset(candidate_sources):
        return False
    return all(_source_is_grounded(source, context) for source in candidate.sources)


def _source_is_grounded(source: SourceRef, context: WikiContext) -> bool:
    if source.kind == "workspace-file":
        return True
    if source.kind == "conversation":
        return context.user_confirmed and source.session_id == context.session_id
    return context.reliable_source


def _pages_are_safe(candidate: WikiCandidate) -> bool:
    try:
        for change in candidate.pages:
            render_page(change.page)
    except (UnsafeWikiPage, ValueError):
        return False
    return True


def _summary_is_safe(candidate: WikiCandidate) -> bool:
    if contains_sensitive_text(candidate.summary):
        return False
    # Reuse the canonical page validator so audit summaries cannot smuggle local
    # paths, credential URLs, or secret headers that page bodies reject.
    sample = candidate.pages[0].page.model_copy(update={"body": f"{candidate.summary}\n"})
    try:
        render_page(sample)
    except (UnsafeWikiPage, ValueError):
        return False
    return True


def _content_fingerprint(page: WikiPage) -> tuple[object, ...]:
    return (
        page.title.casefold(),
        tuple(sorted(tag.casefold() for tag in page.tags)),
        page.body.strip(),
    )


def _knowledge_fingerprint(page: WikiPage) -> str:
    return " ".join(page.body.casefold().split())


def _source_key(source: SourceRef) -> str:
    return source.model_dump_json(exclude_none=True)
