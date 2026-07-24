from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from kosong.tooling import DisplayBlock
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kimi_cli.wiki.manager import PreparedWikiChange

_MAX_WIKI_APPROVAL_ITEMS = 20
_MAX_WIKI_APPROVAL_ITEM_CHARS = 160
_MAX_WIKI_APPROVAL_ITEM_BYTES = 256
_MAX_WIKI_APPROVAL_SUMMARY_CHARS = 240
_MAX_WIKI_APPROVAL_SUMMARY_BYTES = 512
_MAX_WIKI_APPROVAL_BYTES = 8192


class DiffDisplayBlock(DisplayBlock):
    """Display block describing a file diff."""

    type: str = "diff"
    path: str
    old_text: str
    new_text: str
    old_start: int = 1
    new_start: int = 1
    is_summary: bool = False


class TodoDisplayItem(BaseModel):
    title: str
    status: Literal["pending", "in_progress", "done"]


class TodoDisplayBlock(DisplayBlock):
    """Display block describing a todo list update."""

    type: str = "todo"
    items: list[TodoDisplayItem]


class ShellDisplayBlock(DisplayBlock):
    """Display block describing a shell command."""

    type: str = "shell"
    language: str
    command: str


class BackgroundTaskDisplayBlock(DisplayBlock):
    """Display block describing a background task."""

    type: str = "background_task"
    task_id: str
    kind: str
    status: str
    description: str


class WikiApprovalOmitted(BaseModel):
    """Counts excluded from each bounded Wiki approval category."""

    pages: int = 0
    sources: int = 0
    duplicates: int = 0
    conflicts: int = 0


class WikiApprovalBlock(DisplayBlock):
    """Compact, path-safe metadata for a managed Wiki approval."""

    type: str = "wiki"
    summary: str
    pages: list[str]
    sources: list[str] = Field(default_factory=list)
    duplicate_pages: list[str] = Field(default_factory=list)
    conflict_pages: list[str] = Field(default_factory=list)
    workspace_id: str | None
    session_id: str
    details: list[str]
    omitted: WikiApprovalOmitted = Field(default_factory=WikiApprovalOmitted)

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedWikiChange,
        *,
        workspace_id: str | None,
        session_id: str,
    ) -> WikiApprovalBlock:
        raw = {
            "pages": prepared.pages,
            "sources": prepared.source_ids,
            "duplicate_pages": prepared.duplicate_pages,
            "conflict_pages": prepared.conflict_pages,
        }
        visible = {name: _bounded_unique(values) for name, values in raw.items()}

        def build() -> WikiApprovalBlock:
            return cls(
                summary=_bounded_text(
                    " ".join(prepared.summary.split()),
                    char_limit=_MAX_WIKI_APPROVAL_SUMMARY_CHARS,
                    byte_limit=_MAX_WIKI_APPROVAL_SUMMARY_BYTES,
                ),
                pages=visible["pages"],
                sources=visible["sources"],
                duplicate_pages=visible["duplicate_pages"],
                conflict_pages=visible["conflict_pages"],
                workspace_id=(_bounded_text(workspace_id) if workspace_id is not None else None),
                session_id=_bounded_text(session_id),
                details=["Paths are normalized relative to the managed Wiki."],
                omitted=WikiApprovalOmitted(
                    pages=len(raw["pages"]) - len(visible["pages"]),
                    sources=len(raw["sources"]) - len(visible["sources"]),
                    duplicates=(len(raw["duplicate_pages"]) - len(visible["duplicate_pages"])),
                    conflicts=(len(raw["conflict_pages"]) - len(visible["conflict_pages"])),
                ),
            )

        block = build()
        while len(block.model_dump_json().encode("utf-8")) > _MAX_WIKI_APPROVAL_BYTES:
            populated = [name for name, values in visible.items() if values]
            if not populated:
                raise ValueError("Wiki approval metadata cannot fit its byte budget")
            largest = max(
                populated,
                key=lambda name: len(visible[name][-1].encode("utf-8")),
            )
            visible[largest].pop()
            block = build()
        return block


def _bounded_unique(values: Iterable[str]) -> list[str]:
    visible: list[str] = []
    seen: set[str] = set()
    for value in values:
        bounded = _bounded_text(value)
        if bounded in seen:
            continue
        seen.add(bounded)
        visible.append(bounded)
        if len(visible) == _MAX_WIKI_APPROVAL_ITEMS:
            break
    return visible


def _bounded_text(
    value: str,
    *,
    char_limit: int = _MAX_WIKI_APPROVAL_ITEM_CHARS,
    byte_limit: int = _MAX_WIKI_APPROVAL_ITEM_BYTES,
) -> str:
    if len(value) <= char_limit and len(value.encode("utf-8")) <= byte_limit:
        return value
    marker = "…"
    prefix = value[: max(0, char_limit - len(marker))]
    prefix_bytes = prefix.encode("utf-8")[: max(0, byte_limit - len(marker.encode("utf-8")))]
    return prefix_bytes.decode("utf-8", errors="ignore") + marker
