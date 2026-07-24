"""The sole model-facing interface for the shared, managed Wiki."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, override
from uuid import UUID

from kosong.tooling import BriefDisplayBlock, CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.utils.logging import logger
from kimi_cli.wiki.locking import WikiBusyError
from kimi_cli.wiki.models import CurrentSource, WikiCandidate
from kimi_cli.wiki.transaction import WikiConflictError, WikiRecoveryRequired
from kimi_cli.wiki.value_gate import DiscardedCandidate, WikiContext

if TYPE_CHECKING:
    from kimi_cli.soul.agent import Runtime
    from kimi_cli.wiki.manager import PreparedWikiChange, WikiManager


_ARCHIVE_SUFFIXES = frozenset({".7z", ".bz2", ".gz", ".rar", ".tar", ".tgz", ".xz", ".zip"})


class Params(BaseModel):
    """One controlled Wiki operation.

    ``source`` intentionally has no raw-path or URL variant: ingest is limited to
    current-turn inline content or a portable, registry-resolved workspace file.
    """

    operation: Literal["search", "read", "remember", "ingest", "lint"]
    query: str | None = Field(default=None, description="Search query for the global Wiki.")
    page: str | None = Field(
        default=None,
        description="Logical Wiki page path, or a declared category when linting one category.",
    )
    candidate: WikiCandidate | None = Field(
        default=None,
        description="Structured, sourced high-value change proposal for remember or ingest.",
    )
    source: CurrentSource | None = Field(
        default=None,
        description="Current-turn inline content or a registered workspace-relative file only.",
    )
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of search results.")


@dataclass(frozen=True, slots=True)
class WikiToolContext:
    """Trusted per-turn admission facts supplied by runtime wiring in Task 9.

    The model never provides this object.  It separates stable provenance and
    source permissions from the untrusted structured candidate payload.
    """

    provenance_session_id: UUID
    conversation_hashes: frozenset[str]
    allowed_workspace_ids: frozenset[UUID]
    candidate_high_value: bool
    stable: bool
    user_confirmed: bool
    reliable_source: bool


class Wiki(CallableTool2[Params]):
    """Search and read global knowledge; prepare but never directly write changes.

    Task 10 adds the approval-and-commit boundary. Keeping this task preparation
    only makes a tool call unable to bypass that future permission boundary.
    """

    name = "Wiki"
    description = (
        "Search and read the global user Wiki, or prepare a sourced durable knowledge "
        "proposal. Use this tool instead of normal file mutation tools for Wiki content. "
        "Ingest accepts only current-turn inline content or a registered workspace-relative file."
    )
    params = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._runtime = runtime

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        manager = getattr(self._runtime, "wiki", None)
        if manager is None:
            return ToolError(
                message="Global Wiki is unavailable for this session.",
                brief="Wiki unavailable",
            )
        try:
            if params.operation == "search":
                return await self._search(manager, params)
            if params.operation == "read":
                return await self._read(manager, params)
            if params.operation == "lint":
                return await self._lint(manager, params)
            return await self._prepare(manager, params)
        except (OSError, ValueError, UnicodeError) as exc:
            logger.warning("Wiki operation failed: {error}", error=exc)
            return ToolError(
                message="Wiki operation failed. Check the request and try again.",
                brief="Wiki operation failed",
            )
        except (WikiBusyError, WikiConflictError, WikiRecoveryRequired) as exc:
            logger.warning("Wiki operation requires retry: {error}", error=exc)
            return ToolError(
                message="Wiki changed or is busy. Refresh the Wiki state and retry.",
                brief="Wiki retry required",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected Wiki operation failure")
            return ToolError(
                message="Wiki operation failed. Check the request and try again.",
                brief="Wiki operation failed",
            )

    async def _search(self, manager: WikiManager, params: Params) -> ToolReturnValue:
        query = (params.query or "").strip()
        if not query:
            return ToolError(message="Wiki search requires a query.", brief="Missing Wiki query")
        results = await asyncio.to_thread(manager.search, query, params.limit)
        return _ok(
            {
                "results": [
                    {
                        "path": item.logical_path,
                        "title": item.title,
                        "summary": item.summary,
                        "snippet": item.snippet,
                        "score": item.score,
                        "revision": item.revision,
                    }
                    for item in results
                ]
            },
            brief=f"Wiki search: {len(results)} result(s)",
        )

    async def _read(self, manager: WikiManager, params: Params) -> ToolReturnValue:
        if not params.page:
            return ToolError(
                message="Wiki read requires a logical page path.",
                brief="Missing Wiki page",
            )
        result = await asyncio.to_thread(manager.read, params.page)
        return _ok(
            {
                "page": result.page.logical_path,
                "title": result.page.title,
                "revision": result.page.revision,
                "global_revision": result.global_revision,
                "content": result.content,
            },
            brief=f"Read Wiki page: {result.page.logical_path}",
        )

    async def _lint(self, manager: WikiManager, params: Params) -> ToolReturnValue:
        report = await asyncio.to_thread(manager.lint, params.page)
        return _ok(
            {
                "scope": report.scope,
                "scanned_pages": report.scanned_pages,
                "issues": [
                    {
                        "code": issue.code,
                        "page": issue.logical_path,
                        "detail": issue.detail,
                        "related_pages": list(issue.related_paths),
                    }
                    for issue in report.issues
                ],
            },
            brief=f"Wiki lint: {len(report.issues)} issue(s)",
        )

    async def _prepare(self, manager: WikiManager, params: Params) -> ToolReturnValue:
        if params.candidate is None:
            return ToolError(
                message=f"Wiki {params.operation} requires a structured candidate.",
                brief="Missing Wiki candidate",
            )
        operation = cast(Literal["remember", "ingest"], params.operation)
        context = self._context(operation, params.candidate, params.source)
        if isinstance(context, ToolError):
            return context
        prepared: PreparedWikiChange | DiscardedCandidate
        if params.operation == "remember":
            prepared = await asyncio.to_thread(manager.prepare, params.candidate, context)
        else:
            if params.source is None:
                return ToolError(
                    message="Wiki ingest requires current-turn source content.",
                    brief="Missing Wiki source",
                )
            if _is_archive_source(params.source):
                return ToolError(
                    message="Wiki ingest does not accept archive sources.",
                    brief="Unsupported Wiki source",
                )
            prepared = await asyncio.to_thread(
                manager.ingest,
                params.source,
                params.candidate,
                context,
            )
        if isinstance(prepared, DiscardedCandidate):
            return ToolError(
                message=f"Wiki candidate discarded: {prepared.reason}.",
                brief="Wiki candidate discarded",
            )
        return _ok(
            {
                "status": "prepared",
                "summary": prepared.summary,
                "pages": list(prepared.pages),
            },
            brief=f"Wiki proposal prepared: {len(prepared.pages)} page(s)",
        )

    def _context(
        self,
        operation: Literal["remember", "ingest"],
        candidate: WikiCandidate,
        source: CurrentSource | None,
    ) -> WikiContext | ToolError:
        trusted = getattr(self._runtime, "wiki_tool_context", None)
        if not isinstance(trusted, WikiToolContext):
            return ToolError(
                message=(
                    "Wiki write proposal is unavailable until trusted session context is ready."
                ),
                brief="Wiki context unavailable",
            )
        if not trusted.candidate_high_value or not trusted.stable:
            return ToolError(
                message="Wiki candidate lacks trusted high-value or stability evidence.",
                brief="Wiki candidate discarded",
            )
        if (
            source is not None
            and source.kind == "workspace-file"
            and source.workspace_id not in trusted.allowed_workspace_ids
        ):
            return ToolError(
                message="Wiki ingest source is outside the trusted allowed workspace.",
                brief="Wiki candidate discarded",
            )
        if not _sources_are_trusted(candidate, source, trusted):
            return ToolError(
                message="Wiki candidate is not grounded in this session's trusted sources.",
                brief="Wiki candidate discarded",
            )
        return WikiContext(
            session_id=trusted.provenance_session_id,
            cross_turn_utility=trusted.candidate_high_value,
            stable=trusted.stable,
            user_confirmed=trusted.user_confirmed,
            reliable_source=trusted.reliable_source,
            operation=operation,
        )


def _ok(payload: object, *, brief: str) -> ToolReturnValue:
    return ToolReturnValue(
        is_error=False,
        output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        message="",
        display=[BriefDisplayBlock(text=brief)],
    )


def _is_archive_source(source: CurrentSource) -> bool:
    if source.kind != "workspace-file" or source.relative_path is None:
        return False
    return any(source.relative_path.casefold().endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def _sources_are_trusted(
    candidate: WikiCandidate,
    current_source: CurrentSource | None,
    context: WikiToolContext,
) -> bool:
    """Verify all supplied provenance against the trusted current-turn context."""
    if (
        current_source is not None
        and current_source.kind == "workspace-file"
        and current_source.workspace_id not in context.allowed_workspace_ids
    ):
        return False
    sources = (
        *candidate.sources,
        *(source for page in candidate.pages for source in page.page.sources),
    )
    for source in sources:
        if source.kind == "conversation":
            if (
                source.session_id != context.provenance_session_id
                or source.content_hash not in context.conversation_hashes
            ):
                return False
        elif source.kind == "workspace-file":
            if source.workspace_id not in context.allowed_workspace_ids:
                return False
        elif not context.reliable_source:
            return False
    return True
