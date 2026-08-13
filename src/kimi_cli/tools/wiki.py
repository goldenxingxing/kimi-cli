"""The sole model-facing interface for the shared, managed Wiki."""

import asyncio
import json
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Literal, cast, override
from uuid import UUID

from kosong.tooling import BriefDisplayBlock, CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.display import WikiApprovalBlock
from kimi_cli.utils.logging import logger
from kimi_cli.wiki.locking import WikiBusyError
from kimi_cli.wiki.manager import PreparedWikiChange, WikiManager
from kimi_cli.wiki.models import CurrentSource, SourceRef, WikiCandidate, has_url_credentials
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.telemetry import track_wiki_event
from kimi_cli.wiki.transaction import WikiConflictError, WikiRecoveryRequired
from kimi_cli.wiki.triggers import (
    CheckpointDiscardReason,
    WikiAdmissionGrant,
    WikiCheckpoint,
)
from kimi_cli.wiki.value_gate import DiscardedCandidate, WikiContext

_ARCHIVE_SUFFIXES = frozenset({".7z", ".bz2", ".gz", ".rar", ".tar", ".tgz", ".xz", ".zip"})
_EXPLICIT_REMEMBER_INTENT = re.compile(
    r"(?is)(?:"
    r"\bremember\s+(?:this|that|the\s+following)\b|"
    r"\b(?:record|save|store|add)\b.{0,48}\b(?:wiki|memory|knowledge)\b|"
    r"\b(?:wiki|memory|knowledge)\b.{0,48}\b(?:record|save|store|add)\b|"
    r"(?:记住|记录|保存|写入|添加).{0,32}(?:wiki|知识库|记忆|这(?:个|些)?|此)|"
    r"(?:wiki|知识库|记忆).{0,32}(?:记录|保存|写入|添加)"
    r")"
)
_EXPLICIT_REMEMBER_NEGATION = re.compile(
    r"(?is)(?:"
    r"\b(?:do\s+not|don't|never)\b.{0,16}\b(?:remember|record|save|store|add)\b|"
    r"(?:不要|别|无需|不必).{0,8}(?:记住|记录|保存|写入|添加)"
    r")"
)


WikiDiscardReason = Literal[
    "low_value",
    "unstable",
    "ungrounded",
    "duplicate",
    "not_reusable",
]

_DISCARD_REASON_TO_TERMINAL: dict[WikiDiscardReason, CheckpointDiscardReason] = {
    "low_value": "not_useful",
    "unstable": "not_useful",
    "ungrounded": "not_useful",
    "duplicate": "superseded",
    "not_reusable": "not_useful",
}


class Params(BaseModel):
    """One controlled Wiki operation.

    ``source`` intentionally has no raw-path or URL variant: ingest is limited to
    current-turn inline content or a portable, registry-resolved workspace file.
    """

    operation: Literal["search", "read", "remember", "ingest", "lint", "discard"]
    checkpoint_id: str | None = Field(
        default=None,
        description=(
            "The runtime-issued checkpoint this call resolves. Required for remember, "
            "ingest, and discard. It must be copied from a checkpoint block; it cannot "
            "be invented."
        ),
    )
    discard_reason: WikiDiscardReason | None = Field(
        default=None,
        description="Why the checkpoint is not worth persisting. Required for discard.",
    )
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
    explicit_remember_intent: bool = False


_current_wiki_turn_context = ContextVar[WikiToolContext | None](
    "current_wiki_turn_context",
    default=None,
)


def set_wiki_turn_context(
    runtime: Runtime,
    user_text: str,
    *,
    trusted_user_input: bool,
) -> Token[WikiToolContext | None]:
    """Install ephemeral write evidence for one real user turn.

    Synthetic/internal prompts keep the fail-closed base context. Only hashes
    of current-turn text are retained; raw conversation content is never added
    to runtime state or Wiki metadata.
    """
    base = getattr(runtime, "wiki_tool_context", None)
    if not trusted_user_input or not isinstance(base, WikiToolContext):
        return _current_wiki_turn_context.set(None)
    normalized = user_text.strip()
    explicit_remember_intent = _has_explicit_remember_intent(normalized)
    hashes = set(base.conversation_hashes)
    if user_text:
        hashes.add(content_hash(user_text.encode("utf-8")))
    if normalized:
        hashes.add(content_hash(normalized.encode("utf-8")))
    return _current_wiki_turn_context.set(
        WikiToolContext(
            provenance_session_id=base.provenance_session_id,
            conversation_hashes=frozenset(hashes),
            allowed_workspace_ids=base.allowed_workspace_ids,
            candidate_high_value=base.candidate_high_value,
            stable=base.stable,
            user_confirmed=base.user_confirmed or explicit_remember_intent,
            reliable_source=base.reliable_source,
            explicit_remember_intent=(base.explicit_remember_intent or explicit_remember_intent),
        )
    )


def reset_wiki_turn_context(token: Token[WikiToolContext | None]) -> None:
    _current_wiki_turn_context.reset(token)


def extend_wiki_turn_context(user_text: str) -> None:
    """Add trusted steer text hashes to an already active real user turn."""
    active = _current_wiki_turn_context.get()
    if active is None:
        return
    hashes = set(active.conversation_hashes)
    normalized = user_text.strip()
    explicit_remember_intent = _has_explicit_remember_intent(normalized)
    if user_text:
        hashes.add(content_hash(user_text.encode("utf-8")))
    if normalized:
        hashes.add(content_hash(normalized.encode("utf-8")))
    _current_wiki_turn_context.set(
        replace(
            active,
            conversation_hashes=frozenset(hashes),
            user_confirmed=active.user_confirmed or explicit_remember_intent,
            explicit_remember_intent=(active.explicit_remember_intent or explicit_remember_intent),
        )
    )


class Wiki(CallableTool2[Params]):
    """Search/read global knowledge and gate every managed write."""

    name = "Wiki"
    description = (
        "Search and read the global user Wiki, or prepare a sourced durable knowledge "
        "proposal. Use this tool instead of normal file mutation tools for Wiki content. "
        "Ingest accepts only current-turn inline content or a registered workspace-relative file."
    )
    params = Params

    def __init__(self, runtime: Runtime) -> None:
        trusted = getattr(runtime, "wiki_tool_context", None)
        provenance_help = ""
        if isinstance(trusted, WikiToolContext):
            provenance_help = (
                " For conversation SourceRef provenance, use session_id "
                f"{trusted.provenance_session_id} and the SHA-256 hash of the exact "
                "trusted current-turn text."
            )
            if runtime.workspace_id is not None:
                provenance_help += f" The current portable workspace_id is {runtime.workspace_id}."
        super().__init__(description=self.description + provenance_help)
        self._runtime = runtime

    @staticmethod
    def current_context(runtime: Runtime) -> WikiToolContext | None:
        base = getattr(runtime, "wiki_tool_context", None)
        active = _current_wiki_turn_context.get()
        if not isinstance(base, WikiToolContext):
            return None
        if (
            isinstance(active, WikiToolContext)
            and active.provenance_session_id == base.provenance_session_id
        ):
            return active
        return base

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        if getattr(self._runtime, "role", "root") != "root":
            # Defence in depth: a custom agent spec may still list Wiki.
            return ToolError(
                message="The global Wiki is available to the root agent only.",
                brief="Wiki is root-only",
            )
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
            if params.operation == "discard":
                return await self._discard(params)
            return await self._write(manager, params)
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
        except Exception as exc:
            logger.exception("Unexpected Wiki operation failure")
            track_wiki_event(
                "wiki_trigger_failed",
                stage=f"wiki_{params.operation}",
                error_class=type(exc).__name__,
            )
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

    async def _discard(self, params: Params) -> ToolReturnValue:
        """Close a checkpoint the root judged not worth persisting.

        Nothing is written, prepared, or queued: a discard is purely the root
        declining an opportunity, and it is the only alternative to persisting.
        """
        coordinator = getattr(self._runtime, "wiki_coordinator", None)
        if coordinator is None:
            return ToolError(
                message="There is no open checkpoint to discard.",
                brief="No Wiki checkpoint",
            )
        if params.discard_reason is None:
            return ToolError(
                message="Wiki discard requires a discard_reason.",
                brief="Missing discard reason",
            )
        checkpoint_id = (params.checkpoint_id or "").strip()
        checkpoint = await coordinator.resolvable_checkpoint(checkpoint_id)
        if checkpoint is None:
            return ToolError(
                message=(
                    "That checkpoint is not open for resolution. Use a checkpoint_id from "
                    "the current checkpoint block."
                ),
                brief="Unknown Wiki checkpoint",
            )
        await coordinator.discard(
            checkpoint.checkpoint_id,
            _DISCARD_REASON_TO_TERMINAL[params.discard_reason],
        )
        return _ok(
            {"checkpoint_id": checkpoint.checkpoint_id, "resolution": "discarded"},
            brief="Wiki checkpoint discarded",
        )

    async def _resolvable_checkpoint_or_error(self, params: Params) -> object:
        """Require a real, open checkpoint before any write is even prepared."""
        coordinator = getattr(self._runtime, "wiki_coordinator", None)
        if coordinator is None:
            return None
        checkpoint_id = (params.checkpoint_id or "").strip()
        if not checkpoint_id:
            return ToolError(
                message=(
                    f"Wiki {params.operation} requires the checkpoint_id it resolves. "
                    "Copy it from the current checkpoint block, or call "
                    'Wiki(operation="discard") instead.'
                ),
                brief="Missing Wiki checkpoint",
            )
        checkpoint = await coordinator.resolvable_checkpoint(checkpoint_id)
        if checkpoint is None:
            return ToolError(
                message=(
                    "That checkpoint is not open for resolution. Use a checkpoint_id from "
                    "the current checkpoint block."
                ),
                brief="Unknown Wiki checkpoint",
            )
        return checkpoint

    async def _close_checkpoint(
        self,
        checkpoint: object,
        reason: CheckpointDiscardReason,
    ) -> None:
        coordinator = getattr(self._runtime, "wiki_coordinator", None)
        if coordinator is None or not isinstance(checkpoint, WikiCheckpoint):
            return
        await coordinator.discard(checkpoint.checkpoint_id, reason)

    async def _consume_checkpoint(self, checkpoint: object) -> None:
        coordinator = getattr(self._runtime, "wiki_coordinator", None)
        if coordinator is None or not isinstance(checkpoint, WikiCheckpoint):
            return
        await coordinator.consume_checkpoint(checkpoint.checkpoint_id)

    async def _fill_sources_from_checkpoint(
        self,
        checkpoint: WikiCheckpoint,
        candidate: WikiCandidate,
    ) -> WikiCandidate:
        """Supply provenance the runtime observed, when the model supplied none.

        Task 6 established that a model-supplied source hash contributes
        nothing to authorization — only the runtime's own evidence does. So
        demanding that the model restate those hashes was pure friction: it
        would have to guess the workspace id and reproduce a hash byte for
        byte, which is not something it can do. The model supplies the
        content; the runtime supplies the provenance.

        A candidate that does name its own sources is left exactly as it is,
        and still faces the full check.
        """
        coordinator = getattr(self._runtime, "wiki_coordinator", None)
        if coordinator is None:
            return candidate
        needs_fill = not candidate.sources or any(
            not change.page.sources for change in candidate.pages
        )
        if not needs_fill:
            return candidate
        sources = await coordinator.checkpoint_sources(checkpoint.checkpoint_id)
        if not sources:
            return candidate
        pages = [
            change.model_copy(
                update={"page": change.page.model_copy(update={"sources": list(sources)})}
            )
            if not change.page.sources
            else change
            for change in candidate.pages
        ]
        return candidate.model_copy(
            update={
                "sources": list(candidate.sources) or list(sources),
                "pages": pages,
            }
        )

    async def _reserve_grant(
        self,
        manager: WikiManager,
        checkpoint: WikiCheckpoint,
        params: Params,
        candidate_hash: str,
    ) -> WikiAdmissionGrant | None:
        coordinator = getattr(self._runtime, "wiki_coordinator", None)
        assert params.candidate is not None
        if coordinator is None:
            return None
        if (
            params.source is not None
            and params.source.kind == "workspace-file"
            and params.source.workspace_id != self._runtime.workspace_id
        ):
            return None
        trusted = self.current_context(self._runtime)
        conversation_hashes: frozenset[str] = (
            trusted.conversation_hashes if trusted is not None else frozenset[str]()
        )
        session_id = (
            trusted.provenance_session_id
            if trusted is not None
            else coordinator.provenance_session_id
        )
        verified = await asyncio.to_thread(
            _verified_source_keys,
            manager,
            params.candidate,
            conversation_hashes,
            session_id,
        )
        return await coordinator.reserve_grant(
            checkpoint.checkpoint_id,
            candidate_hash=candidate_hash,
            source_keys=_candidate_source_keys(params.candidate),
            verified_source_keys=verified,
        )

    async def _finish_grant(
        self,
        grant: WikiAdmissionGrant | None,
        outcome: Literal["persisted", "declined", "discarded", "failed"],
    ) -> None:
        coordinator = getattr(self._runtime, "wiki_coordinator", None)
        if coordinator is None or grant is None:
            return
        await coordinator.finish_grant(
            grant.checkpoint_id,
            outcome=outcome,
            candidate_hash=grant.candidate_hash,
        )

    async def _write(self, manager: WikiManager, params: Params) -> ToolReturnValue:
        resolvable = await self._resolvable_checkpoint_or_error(params)
        if isinstance(resolvable, ToolError):
            return resolvable
        if params.candidate is None:
            return ToolError(
                message=f"Wiki {params.operation} requires a structured candidate.",
                brief="Missing Wiki candidate",
            )
        operation = cast(Literal["remember", "ingest"], params.operation)
        grant: WikiAdmissionGrant | None = None
        if isinstance(resolvable, WikiCheckpoint):
            params = params.model_copy(
                update={
                    "candidate": await self._fill_sources_from_checkpoint(
                        resolvable, params.candidate
                    )
                }
            )
            assert params.candidate is not None
            grant = await self._reserve_grant(
                manager, resolvable, params, _candidate_hash(params.candidate)
            )
            if grant is None:
                return ToolError(
                    message=(
                        "This candidate is not admissible: its sources do not match what the "
                        "runtime actually observed for that checkpoint. Re-read the exact "
                        "source and try again, or discard the checkpoint."
                    ),
                    brief="Wiki candidate not admitted",
                )
            context = WikiContext.from_grant(grant, operation)
        else:
            legacy = self._context(operation, params.candidate, params.source)
            if isinstance(legacy, ToolError):
                return legacy
            context = legacy
        try:
            return await self._prepare_and_commit(
                manager, params, context, resolvable=resolvable, grant=grant
            )
        except (WikiBusyError, WikiConflictError, WikiRecoveryRequired):
            # The Wiki moved under us. Hand the checkpoint back so the identical
            # candidate may retry after a fresh read; changed content must earn
            # a new grant.
            await self._release_or_finish(grant)
            raise
        except BaseException:
            # No path may leave write authority outstanding, including a
            # cancellation or an unexpected failure inside Approval.
            await self._finish_grant(grant, "failed")
            raise

    async def _release_or_finish(self, grant: WikiAdmissionGrant | None) -> None:
        """Release a grant for one retry, or spend it if it cannot be released."""
        coordinator = getattr(self._runtime, "wiki_coordinator", None)
        if coordinator is None or grant is None:
            return
        if not await coordinator.release_retry(grant.checkpoint_id, grant.candidate_hash):
            await self._finish_grant(grant, "failed")

    async def _prepare_and_commit(
        self,
        manager: WikiManager,
        params: Params,
        context: WikiContext,
        *,
        resolvable: object,
        grant: WikiAdmissionGrant | None,
    ) -> ToolReturnValue:
        assert params.candidate is not None
        prepared: PreparedWikiChange | DiscardedCandidate
        if params.operation == "remember":
            prepared = await asyncio.to_thread(manager.prepare, params.candidate, context)
        else:
            if params.source is None:
                await self._finish_grant(grant, "failed")
                return ToolError(
                    message="Wiki ingest requires current-turn source content.",
                    brief="Missing Wiki source",
                )
            if _is_archive_source(params.source):
                await self._finish_grant(grant, "failed")
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
            # The gate already decided; the opportunity is spent either way.
            await self._finish_grant(grant, "discarded")
            track_wiki_event(
                "wiki_candidate_discarded",
                reason=prepared.reason,
                checkpoint_id=_checkpoint_id_of(resolvable),
                page_count=len(params.candidate.pages),
            )
            await self._close_checkpoint(resolvable, "not_useful")
            return ToolError(
                message=f"Wiki candidate discarded: {prepared.reason}.",
                brief="Wiki candidate discarded",
            )
        approval = self._runtime.approval
        if not approval.is_yolo():
            trusted = self.current_context(self._runtime)
            assert trusted is not None
            track_wiki_event(
                "wiki_approval_requested",
                mode="afk" if approval.is_afk() else "normal",
                page_count=len(prepared.pages),
                checkpoint_id=_checkpoint_id_of(resolvable),
            )
            result = await approval.request(
                self.name,
                "wiki.write",
                f"Record: {prepared.summary}\nChanges: {len(prepared.pages)} pages",
                display=[
                    WikiApprovalBlock.from_prepared(
                        prepared,
                        workspace_id=(
                            str(self._runtime.workspace_id)
                            if self._runtime.workspace_id is not None
                            else None
                        ),
                        session_id=str(trusted.provenance_session_id),
                    )
                ],
                request_policy="session_only",
            )
            if not result:
                await self._finish_grant(grant, "declined")
                await self._close_checkpoint(resolvable, "user_declined")
                # Same as memory: recording a page is incidental to whatever
                # the agent was actually asked to do, so declining it must not
                # end the turn.
                return result.rejection_error(stops_turn=False)
        committed = await asyncio.to_thread(manager.commit, prepared)
        track_wiki_event(
            "wiki_committed",
            page_count=len(committed.pages),
            global_revision=committed.global_revision,
            checkpoint_id=_checkpoint_id_of(resolvable),
        )
        await self._finish_grant(grant, "persisted")
        await self._consume_checkpoint(resolvable)
        return _ok(
            {
                "status": "committed",
                "summary": prepared.summary,
                "pages": list(committed.pages),
                "global_revision": committed.global_revision,
                "search_index_current": committed.search_index_current,
            },
            brief=f"Wiki updated: {len(committed.pages)} page(s)",
        )

    def _context(
        self,
        operation: Literal["remember", "ingest"],
        candidate: WikiCandidate,
        source: CurrentSource | None,
    ) -> WikiContext | ToolError:
        trusted = self.current_context(self._runtime)
        if not isinstance(trusted, WikiToolContext):
            return ToolError(
                message=(
                    "Wiki write proposal is unavailable until trusted session context is ready."
                ),
                brief="Wiki context unavailable",
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
        source_evidence = _has_independent_source_evidence(candidate, trusted)
        cross_turn_utility = (
            trusted.candidate_high_value or trusted.explicit_remember_intent or source_evidence
        )
        stable = trusted.stable or trusted.explicit_remember_intent or source_evidence
        if not cross_turn_utility or not stable:
            return ToolError(
                message="Wiki candidate lacks trusted high-value or stability evidence.",
                brief="Wiki candidate discarded",
            )
        return WikiContext(
            session_id=trusted.provenance_session_id,
            cross_turn_utility=cross_turn_utility,
            stable=stable,
            user_confirmed=trusted.user_confirmed,
            reliable_source=trusted.reliable_source,
            operation=operation,
        )


def _checkpoint_id_of(checkpoint: object) -> str | None:
    return checkpoint.checkpoint_id if isinstance(checkpoint, WikiCheckpoint) else None


def _candidate_hash(candidate: WikiCandidate) -> str:
    """Identify the exact proposal a grant authorizes, content and all."""
    return content_hash(
        candidate.model_dump_json(exclude_none=True, round_trip=True).encode("utf-8")
    )


def _candidate_source_keys(candidate: WikiCandidate) -> frozenset[str]:
    """Every source the candidate and its pages claim, as canonical keys."""
    return frozenset(
        _canonical_source_key(source)
        for source in (
            *candidate.sources,
            *(source for change in candidate.pages for source in change.page.sources),
        )
    )


def _canonical_source_key(source: SourceRef) -> str:
    return source.model_dump_json(exclude_none=True)


def _verified_source_keys(
    manager: WikiManager,
    candidate: WikiCandidate,
    conversation_hashes: frozenset[str],
    session_id: UUID,
) -> frozenset[str]:
    """Re-derive each claimed source and keep only those that still match.

    A source hash the model repeats back proves nothing: workspace files are
    re-read through the registry and re-hashed, web sources must be
    credential-free, and conversation sources must name this session and text
    the runtime actually accepted.
    """
    verified: set[str] = set()
    for source in (
        *candidate.sources,
        *(source for change in candidate.pages for source in change.page.sources),
    ):
        key = _canonical_source_key(source)
        if source.kind == "workspace-file":
            try:
                resolved = manager.registry.resolve(source)
            except (OSError, ValueError):
                continue
            if resolved is None:
                continue
            try:
                current = content_hash(resolved.read_bytes())
            except OSError:
                continue
            if current == source.content_hash:
                verified.add(key)
        elif source.kind == "web":
            if source.url is not None and not has_url_credentials(str(source.url)):
                verified.add(key)
        elif source.kind == "conversation":
            if source.session_id == session_id and source.content_hash in conversation_hashes:
                verified.add(key)
    return frozenset(verified)


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


def _has_explicit_remember_intent(user_text: str) -> bool:
    """Recognize a narrow user-authored request to persist durable knowledge."""
    return bool(
        user_text
        and not _EXPLICIT_REMEMBER_NEGATION.search(user_text)
        and _EXPLICIT_REMEMBER_INTENT.search(user_text)
    )


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


def _has_independent_source_evidence(
    candidate: WikiCandidate,
    context: WikiToolContext,
) -> bool:
    """Derive durable-source evidence without trusting candidate value labels."""
    sources = (
        *candidate.sources,
        *(source for page in candidate.pages for source in page.page.sources),
    )
    return bool(sources) and all(
        (source.kind == "workspace-file" and source.workspace_id in context.allowed_workspace_ids)
        or (source.kind == "web" and context.reliable_source)
        for source in sources
    )
