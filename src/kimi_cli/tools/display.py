from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from kosong.tooling import DisplayBlock
from pydantic import BaseModel

if TYPE_CHECKING:
    from kimi_cli.wiki.manager import PreparedWikiChange

_MAX_WIKI_APPROVAL_PAGES = 20


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


class WikiApprovalBlock(DisplayBlock):
    """Compact, path-safe metadata for a managed Wiki approval."""

    type: str = "wiki"
    summary: str
    pages: list[str]
    workspace_id: str | None
    session_id: str
    details: list[str]

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedWikiChange,
        *,
        workspace_id: str | None,
        session_id: str,
    ) -> WikiApprovalBlock:
        details: list[str] = []
        if prepared.source_ids:
            details.append(f"Sources: {', '.join(prepared.source_ids)}")
        details.append("Paths are normalized relative to the managed Wiki.")
        if prepared.duplicate_pages:
            details.append(f"Duplicates omitted: {', '.join(prepared.duplicate_pages)}")
        if prepared.conflict_pages:
            details.append(f"Conflicts preserved: {', '.join(prepared.conflict_pages)}")
        visible_pages = list(prepared.pages[:_MAX_WIKI_APPROVAL_PAGES])
        omitted_pages = len(prepared.pages) - len(visible_pages)
        if omitted_pages:
            details.append(f"Additional pages omitted: {omitted_pages}.")
        return cls(
            summary=prepared.summary,
            pages=visible_pages,
            workspace_id=workspace_id,
            session_id=session_id,
            details=details,
        )
