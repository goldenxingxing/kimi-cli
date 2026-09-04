"""Tests for single-use, runtime-authenticated Wiki write admission."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.wiki import Params, Wiki, WikiToolContext, set_wiki_turn_context
from kimi_cli.wiki.evidence import WikiEvidenceReporter
from kimi_cli.wiki.intent import detect_durable_intent
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wiki.models import PageChange, SourceRef, WikiCandidate, WikiPage
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.triggers import EvidenceObservation, WikiTurnCoordinator

_NOW = datetime(2026, 8, 3, tzinfo=UTC)


@pytest.fixture
def admission_runtime(runtime: Runtime, tmp_path: Path):
    manager = WikiManager(tmp_path / "wiki", wal=False)
    workspace = Path(str(runtime.session.work_dir)).resolve()
    workspace_id = manager.registry.register(workspace)
    coordinator = WikiTurnCoordinator(
        provenance_session_id=uuid4(),
        workspace_id=workspace_id,
    )
    runtime.wiki = manager
    runtime.workspace_id = workspace_id
    runtime.wiki_coordinator = coordinator
    runtime.wiki_evidence_reporter = WikiEvidenceReporter(coordinator, runtime)
    # Deliberately maximally permissive: the pre-Task-6 turn context would let
    # every candidate below through, so anything these tests reject is rejected
    # by the runtime grant and by nothing else.
    runtime.wiki_tool_context = WikiToolContext(
        provenance_session_id=coordinator.provenance_session_id,
        conversation_hashes=frozenset(),
        allowed_workspace_ids=frozenset({workspace_id}),
        candidate_high_value=True,
        stable=True,
        user_confirmed=True,
        reliable_source=True,
        explicit_remember_intent=True,
    )
    try:
        yield runtime
    finally:
        manager.close()


def _revision(runtime: Runtime) -> str:
    return runtime.wiki.layout.revision.read_text(encoding="utf-8")


def _candidate(source: SourceRef, *, body: str = "Signed tags only for every release.\n"):
    page = WikiPage(
        logical_path="concepts/release-rule.md",
        title="Release rule",
        created=_NOW,
        updated=_NOW,
        tags=["release"],
        sources=[source],
        revision=1,
        body=body,
    )
    return WikiCandidate(
        summary="Record the signed-tag release rule",
        pages=[PageChange(page=page, expected_revision=None)],
        sources=[source],
        value="high",
    )


async def _file_checkpoint(runtime: Runtime, *, name: str = "decision.md"):
    """Open a root_evidence checkpoint grounded in a real workspace file."""
    coordinator = runtime.wiki_coordinator
    workspace = Path(str(runtime.session.work_dir))
    path = workspace / name
    path.write_text("durable decision", encoding="utf-8")
    await coordinator.begin_turn("where is the rule", "where is the rule")
    source = runtime.wiki.registry.relative_source(runtime.workspace_id, path)
    evidence = await coordinator.record_evidence(
        EvidenceObservation(
            root_turn_id=coordinator.active_turn_id,
            workspace_id=runtime.workspace_id,
            producer_role="root",
            producer_id=None,
            run_generation=None,
            tool_call_id="call-read",
            source_class="workspace-file",
            request_hash=content_hash(b"read"),
            result_hash=source.content_hash,
            logical_paths=(source.path,),
            source_refs=(source,),
            reliable=True,
            stable_snapshot=True,
            triggering=True,
        )
    )
    assert evidence is not None
    checkpoint = await coordinator.create_checkpoint(
        "root_evidence",
        evidence_ids=(evidence.evidence_id,),
        summary_hash=content_hash(b"the rule lives in decision.md"),
    )
    return checkpoint, source, path


async def _durable_checkpoint(runtime: Runtime, text: str = "Remember this release rule"):
    coordinator = runtime.wiki_coordinator
    await coordinator.begin_turn(text, text)
    intent = detect_durable_intent(text)
    assert intent is not None
    checkpoint = await coordinator.record_durable_intent(intent)
    assert checkpoint is not None
    token = set_wiki_turn_context(runtime, text, trusted_user_input=True)
    return checkpoint, text, token


# ---------------------------------------------------------------------------
# Forgery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_claims_cannot_forge_a_grant(admission_runtime) -> None:
    """value="high", a self-asserted hash, and a chosen checkpoint ID are all inert."""
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    forged_source = SourceRef(
        kind="workspace-file",
        workspace_id=admission_runtime.workspace_id,
        path=source.path,
        content_hash=content_hash(b"content that was never on disk"),
    )
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id="attacker-chosen",
            candidate=_candidate(forged_source),
        )
    )

    assert result.is_error
    assert _revision(admission_runtime) == before
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0
    assert checkpoint.state == "pending"


@pytest.mark.asyncio
async def test_source_hash_the_runtime_never_observed_is_refused(admission_runtime) -> None:
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    unobserved = SourceRef(
        kind="workspace-file",
        workspace_id=admission_runtime.workspace_id,
        path=source.path,
        content_hash=content_hash(b"different bytes"),
    )
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_candidate(unobserved),
        )
    )

    assert result.is_error
    assert _revision(admission_runtime) == before
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0


@pytest.mark.asyncio
async def test_an_unsourced_candidate_cannot_pass_a_checkpoint_with_no_evidence(
    admission_runtime,
) -> None:
    """Filling borrows the checkpoint's own evidence; a checkpoint with none
    lends nothing, so the candidate stays ungrounded and is refused."""
    coordinator = admission_runtime.wiki_coordinator
    await coordinator.begin_turn("remember this rule", "remember this rule")
    checkpoint = await coordinator.create_checkpoint(
        "explicit_user_durable", summary_hash=content_hash(b"a durable rule")
    )
    unsourced = WikiCandidate(
        summary="Record the release rule",
        pages=[
            PageChange(
                page=WikiPage(
                    logical_path="concepts/release-rule.md",
                    title="Release rule",
                    created=_NOW,
                    updated=_NOW,
                    tags=["release"],
                    sources=[],
                    revision=1,
                    body="Signed tags only.\n",
                ),
                expected_revision=None,
            )
        ],
        sources=[],
        value="high",
    )
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=unsourced,
        )
    )

    assert result.is_error
    assert _revision(admission_runtime) == before


@pytest.mark.asyncio
async def test_a_real_current_file_the_runtime_never_read_is_refused(admission_runtime) -> None:
    """The core admission rule: sources must be evidence *of this checkpoint*.

    The candidate below is entirely honest — the file exists, is inside the
    workspace, and its hash is current, so both the pre-Task-6 turn context and
    the manager's own source resolution accept it. It is refused solely because
    the runtime never observed that file for this checkpoint.
    """
    checkpoint, _, _ = await _file_checkpoint(admission_runtime)
    unread = Path(str(admission_runtime.session.work_dir)) / "never-read.md"
    unread.write_text("a file the agent never opened", encoding="utf-8")
    unread_source = admission_runtime.wiki.registry.relative_source(
        admission_runtime.workspace_id, unread
    )
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_candidate(unread_source),
        )
    )

    assert result.is_error
    assert "did not observe" in result.message
    assert _revision(admission_runtime) == before
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0


@pytest.mark.asyncio
async def test_one_unobserved_source_rejects_the_whole_candidate(admission_runtime) -> None:
    """Grounding is all-or-nothing; a good source cannot carry an unobserved one."""
    checkpoint, observed, _ = await _file_checkpoint(admission_runtime)
    extra = Path(str(admission_runtime.session.work_dir)) / "extra.md"
    extra.write_text("also never opened", encoding="utf-8")
    extra_source = admission_runtime.wiki.registry.relative_source(
        admission_runtime.workspace_id, extra
    )
    candidate = _candidate(observed)
    candidate = candidate.model_copy(update={"sources": [observed, extra_source]})
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=candidate,
        )
    )

    assert result.is_error
    assert _revision(admission_runtime) == before


@pytest.mark.asyncio
async def test_a_web_source_the_runtime_never_fetched_is_refused(admission_runtime) -> None:
    checkpoint, _, _ = await _file_checkpoint(admission_runtime)
    web = SourceRef(
        kind="web",
        url="https://example.test/release-policy",
        content_hash=content_hash(b"a page the agent never fetched"),
    )
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_candidate(web),
        )
    )

    assert result.is_error
    assert _revision(admission_runtime) == before


# ---------------------------------------------------------------------------
# Stale sources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_changed_file_fails_stale_and_a_fresh_read_can_retry(admission_runtime) -> None:
    checkpoint, source, path = await _file_checkpoint(admission_runtime)
    tool = Wiki(admission_runtime)
    path.write_text("changed on disk", encoding="utf-8")
    before = _revision(admission_runtime)

    stale = await tool(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_candidate(source),
        )
    )

    assert stale.is_error
    assert _revision(admission_runtime) == before
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0

    # Re-reading the file produces new evidence and a new checkpoint.
    fresh_checkpoint, fresh_source, _ = await _file_checkpoint(
        admission_runtime, name="decision.md"
    )
    fresh = await tool(
        Params(
            operation="remember",
            checkpoint_id=fresh_checkpoint.checkpoint_id,
            candidate=_candidate(fresh_source),
        )
    )

    assert not fresh.is_error
    assert _revision(admission_runtime) != before


# ---------------------------------------------------------------------------
# Single use and concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_checkpoint_admits_at_most_one_write(admission_runtime) -> None:
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    tool = Wiki(admission_runtime)
    params = Params(
        operation="remember",
        checkpoint_id=checkpoint.checkpoint_id,
        candidate=_candidate(source),
    )

    first = await tool(params)
    second = await tool(params)

    assert not first.is_error
    assert second.is_error
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0


@pytest.mark.asyncio
async def test_concurrent_calls_consume_the_checkpoint_once(admission_runtime) -> None:
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    tool = Wiki(admission_runtime)
    params = Params(
        operation="remember",
        checkpoint_id=checkpoint.checkpoint_id,
        candidate=_candidate(source),
    )

    results = await asyncio.gather(tool(params), tool(params))

    assert sum(not result.is_error for result in results) == 1
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0


@pytest.mark.asyncio
async def test_a_declined_approval_spends_the_grant_and_writes_nothing(
    admission_runtime,
) -> None:
    from kimi_cli.soul.approval import Approval

    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    admission_runtime.approval = Approval(yolo=False)
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_candidate(source),
        )
    )

    assert result.is_error
    assert _revision(admission_runtime) == before
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0
    assert (await admission_runtime.wiki_coordinator.pending_batch()).checkpoints == ()


# ---------------------------------------------------------------------------
# Cause-specific grant rules
# ---------------------------------------------------------------------------


def _unsourced(body: str = "Signed tags only for every release.\n") -> WikiCandidate:
    """The candidate a model can actually write: content, and no provenance."""
    page = WikiPage(
        logical_path="concepts/release-rule.md",
        title="Release rule",
        created=_NOW,
        updated=_NOW,
        tags=["release"],
        revision=1,
        body=body,
    )
    return WikiCandidate(
        summary="Record the signed-tag release rule",
        pages=[PageChange(page=page, expected_revision=None)],
        value="high",
    )


@pytest.mark.asyncio
async def test_a_candidate_with_no_sources_is_grounded_by_the_checkpoints_own_evidence(
    admission_runtime,
) -> None:
    """The working path: the model writes the knowledge, the runtime the provenance."""
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_unsourced(),
        )
    )

    assert not result.is_error
    stored = admission_runtime.wiki.read("concepts/release-rule.md").page
    assert stored.sources == [source]
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0


@pytest.mark.asyncio
async def test_an_unsourced_candidate_resolves_an_explicit_durable_checkpoint(
    admission_runtime,
) -> None:
    """A checkpoint with no evidence still has the user's own turn behind it.

    Its source hash is of the exact turn text, which the model cannot compute,
    so the runtime attaches it -- otherwise "remember this" would be a request
    that can only ever be discarded.
    """
    from kimi_cli.tools.wiki import reset_wiki_turn_context

    checkpoint, text, token = await _durable_checkpoint(admission_runtime)
    try:
        result = await Wiki(admission_runtime)(
            Params(
                operation="remember",
                checkpoint_id=checkpoint.checkpoint_id,
                candidate=_unsourced(),
            )
        )

        assert not result.is_error
        stored = admission_runtime.wiki.read("concepts/release-rule.md").page
        assert [source.kind for source in stored.sources] == ["conversation"]
        assert stored.sources[0].content_hash == content_hash(text.encode("utf-8"))
    finally:
        reset_wiki_turn_context(token)


@pytest.mark.asyncio
async def test_an_unobserved_source_is_refused_with_a_recoverable_instruction(
    admission_runtime,
) -> None:
    """A refusal the model cannot act on is how a checkpoint gets burned."""
    checkpoint, _, _ = await _file_checkpoint(admission_runtime)
    unread = Path(str(admission_runtime.session.work_dir)) / "never-read.md"
    unread.write_text("a file the agent never opened", encoding="utf-8")
    unread_source = admission_runtime.wiki.registry.relative_source(
        admission_runtime.workspace_id, unread
    )

    refused = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_candidate(unread_source),
        )
    )

    assert refused.is_error
    assert "Omit `sources`" in refused.message
    # The checkpoint survives the refusal, so following that instruction works.
    accepted = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_unsourced(),
        )
    )

    assert not accepted.is_error


@pytest.mark.asyncio
async def test_a_page_that_cannot_be_stored_as_written_can_be_rewritten(
    admission_runtime,
) -> None:
    """A phrasing the store rejects must not cost the whole checkpoint."""
    checkpoint, _, _ = await _file_checkpoint(admission_runtime)

    refused = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_unsourced(body="The rule lives in /Users/someone/decision.md.\n"),
        )
    )

    assert refused.is_error
    assert "absolute path" in refused.message
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0

    rewritten = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_unsourced(body="The rule lives in the workspace decision note.\n"),
        )
    )

    assert not rewritten.is_error


@pytest.mark.asyncio
async def test_a_body_measured_per_non_ascii_unit_is_not_read_as_a_path(
    admission_runtime,
) -> None:
    """ "O(1)/点" is a unit, not a root directory."""
    checkpoint, _, _ = await _file_checkpoint(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_unsourced(body="流式 O(1)/点，状态 <200 B。\n"),
        )
    )

    assert not result.is_error


@pytest.mark.asyncio
async def test_explicit_durable_intent_admits_its_own_conversation_source(
    admission_runtime,
) -> None:
    from kimi_cli.tools.wiki import reset_wiki_turn_context

    checkpoint, text, token = await _durable_checkpoint(admission_runtime)
    try:
        source = SourceRef(
            kind="conversation",
            session_id=admission_runtime.wiki_coordinator.provenance_session_id,
            content_hash=content_hash(text.encode("utf-8")),
        )

        result = await Wiki(admission_runtime)(
            Params(
                operation="remember",
                checkpoint_id=checkpoint.checkpoint_id,
                candidate=_candidate(source),
            )
        )

        assert not result.is_error
        assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0
    finally:
        reset_wiki_turn_context(token)


@pytest.mark.asyncio
async def test_conversation_source_from_another_turn_is_refused(admission_runtime) -> None:
    from kimi_cli.tools.wiki import reset_wiki_turn_context

    checkpoint, _, token = await _durable_checkpoint(admission_runtime)
    try:
        foreign = SourceRef(
            kind="conversation",
            session_id=admission_runtime.wiki_coordinator.provenance_session_id,
            content_hash=content_hash(b"text the user never said"),
        )
        before = _revision(admission_runtime)

        result = await Wiki(admission_runtime)(
            Params(
                operation="remember",
                checkpoint_id=checkpoint.checkpoint_id,
                candidate=_candidate(foreign),
            )
        )

        assert result.is_error
        assert _revision(admission_runtime) == before
    finally:
        reset_wiki_turn_context(token)


@pytest.mark.asyncio
async def test_conversation_source_cannot_ground_an_evidence_checkpoint(
    admission_runtime,
) -> None:
    """Only explicit user intent may lean on the conversation itself."""
    from kimi_cli.tools.wiki import reset_wiki_turn_context

    checkpoint, _, _ = await _file_checkpoint(admission_runtime)
    text = "the rule lives in decision.md"
    token = set_wiki_turn_context(admission_runtime, text, trusted_user_input=True)
    try:
        conversation = SourceRef(
            kind="conversation",
            session_id=admission_runtime.wiki_coordinator.provenance_session_id,
            content_hash=content_hash(text.encode("utf-8")),
        )
        before = _revision(admission_runtime)

        result = await Wiki(admission_runtime)(
            Params(
                operation="remember",
                checkpoint_id=checkpoint.checkpoint_id,
                candidate=_candidate(conversation),
            )
        )

        assert result.is_error
        assert _revision(admission_runtime) == before
    finally:
        reset_wiki_turn_context(token)


# ---------------------------------------------------------------------------
# Grant lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_retry_only_accepts_the_identical_candidate(admission_runtime) -> None:
    from kimi_cli.tools.wiki import _candidate_hash, _candidate_source_keys, _verified_source_keys

    coordinator = admission_runtime.wiki_coordinator
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    candidate = _candidate(source)
    candidate_hash = _candidate_hash(candidate)
    verified = _verified_source_keys(
        admission_runtime.wiki,
        candidate,
        frozenset[str](),
        coordinator.provenance_session_id,
    )

    grant = await coordinator.reserve_grant(
        checkpoint.checkpoint_id,
        candidate_hash=candidate_hash,
        source_keys=_candidate_source_keys(candidate),
        verified_source_keys=verified,
    )

    assert grant is not None
    assert coordinator.unconsumed_grant_count == 1
    # A different candidate cannot release the reservation...
    assert not await coordinator.release_retry(checkpoint.checkpoint_id, content_hash(b"other"))
    assert coordinator.unconsumed_grant_count == 1
    # ...but the identical one can, returning the checkpoint to pending.
    assert await coordinator.release_retry(checkpoint.checkpoint_id, candidate_hash)
    assert coordinator.unconsumed_grant_count == 0
    assert len((await coordinator.pending_batch()).checkpoints) == 1


@pytest.mark.asyncio
async def test_a_reserved_grant_blocks_a_second_reservation(admission_runtime) -> None:
    from kimi_cli.tools.wiki import _candidate_hash, _candidate_source_keys, _verified_source_keys

    coordinator = admission_runtime.wiki_coordinator
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    candidate = _candidate(source)
    reserve = {
        "candidate_hash": _candidate_hash(candidate),
        "source_keys": _candidate_source_keys(candidate),
        "verified_source_keys": _verified_source_keys(
            admission_runtime.wiki,
            candidate,
            frozenset[str](),
            coordinator.provenance_session_id,
        ),
    }

    first = await coordinator.reserve_grant(checkpoint.checkpoint_id, **reserve)
    second = await coordinator.reserve_grant(checkpoint.checkpoint_id, **reserve)

    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_cancelling_the_turn_invalidates_outstanding_grants(admission_runtime) -> None:
    from kimi_cli.tools.wiki import _candidate_hash, _candidate_source_keys, _verified_source_keys

    coordinator = admission_runtime.wiki_coordinator
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    candidate = _candidate(source)
    await coordinator.reserve_grant(
        checkpoint.checkpoint_id,
        candidate_hash=_candidate_hash(candidate),
        source_keys=_candidate_source_keys(candidate),
        verified_source_keys=_verified_source_keys(
            admission_runtime.wiki,
            candidate,
            frozenset[str](),
            coordinator.provenance_session_id,
        ),
    )

    await coordinator.cancel_turn(checkpoint.root_turn_id)

    assert coordinator.unconsumed_grant_count == 0
    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=candidate,
        )
    )
    assert result.is_error


@pytest.mark.asyncio
async def test_closing_the_runtime_clears_every_grant(admission_runtime) -> None:
    from kimi_cli.tools.wiki import _candidate_hash, _candidate_source_keys, _verified_source_keys

    coordinator = admission_runtime.wiki_coordinator
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    candidate = _candidate(source)
    await coordinator.reserve_grant(
        checkpoint.checkpoint_id,
        candidate_hash=_candidate_hash(candidate),
        source_keys=_candidate_source_keys(candidate),
        verified_source_keys=_verified_source_keys(
            admission_runtime.wiki,
            candidate,
            frozenset[str](),
            coordinator.provenance_session_id,
        ),
    )

    await coordinator.close()

    assert coordinator.unconsumed_grant_count == 0


# ---------------------------------------------------------------------------
# Cross-session isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_another_sessions_coordinator_cannot_admit_this_checkpoint(
    admission_runtime,
    tmp_path: Path,
) -> None:
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    other = WikiTurnCoordinator(
        provenance_session_id=uuid4(),
        workspace_id=admission_runtime.workspace_id,
    )
    await other.begin_turn("other session", "other session")

    grant = await other.reserve_grant(
        checkpoint.checkpoint_id,
        candidate_hash=content_hash(b"candidate"),
        source_keys=frozenset({source.model_dump_json(exclude_none=True)}),
        verified_source_keys=frozenset({source.model_dump_json(exclude_none=True)}),
    )

    assert grant is None
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0
    assert len((await admission_runtime.wiki_coordinator.pending_batch()).checkpoints) == 1


@pytest.mark.asyncio
async def test_grant_records_no_raw_text_or_absolute_path(admission_runtime) -> None:
    from kimi_cli.tools.wiki import _candidate_hash, _candidate_source_keys, _verified_source_keys

    coordinator = admission_runtime.wiki_coordinator
    checkpoint, source, path = await _file_checkpoint(admission_runtime)
    candidate = _candidate(source)

    grant = await coordinator.reserve_grant(
        checkpoint.checkpoint_id,
        candidate_hash=_candidate_hash(candidate),
        source_keys=_candidate_source_keys(candidate),
        verified_source_keys=_verified_source_keys(
            admission_runtime.wiki,
            candidate,
            frozenset[str](),
            coordinator.provenance_session_id,
        ),
    )

    assert grant is not None
    rendered = repr(grant)
    assert str(path) not in rendered
    assert "durable decision" not in rendered
    assert "Signed tags only" not in rendered
    assert isinstance(grant.workspace_id, UUID)


@pytest.mark.asyncio
async def test_a_revision_conflict_returns_the_checkpoint_for_one_retry(
    admission_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retryable conflict is not a spent opportunity."""
    from kimi_cli.wiki.transaction import WikiConflictError

    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    coordinator = admission_runtime.wiki_coordinator
    tool = Wiki(admission_runtime)
    params = Params(
        operation="remember",
        checkpoint_id=checkpoint.checkpoint_id,
        candidate=_candidate(source),
    )
    calls: list[int] = []

    real_commit = admission_runtime.wiki.commit

    def _conflict_once(prepared):
        calls.append(1)
        if len(calls) == 1:
            raise WikiConflictError("the Wiki moved")
        return real_commit(prepared)

    monkeypatch.setattr(admission_runtime.wiki, "commit", _conflict_once)

    first = await tool(params)

    assert first.is_error
    # The grant was released, not spent: the checkpoint is open again.
    assert coordinator.unconsumed_grant_count == 0
    assert len((await coordinator.pending_batch()).checkpoints) == 1

    second = await tool(params)

    assert not second.is_error
    assert coordinator.unconsumed_grant_count == 0
    assert (await coordinator.pending_batch()).checkpoints == ()


@pytest.mark.asyncio
async def test_a_grant_cannot_be_spent_after_its_turn_moved_on(admission_runtime) -> None:
    """An in-flight write must not commit into a turn that already ended."""
    from kimi_cli.tools.wiki import _candidate_hash, _candidate_source_keys, _verified_source_keys

    coordinator = admission_runtime.wiki_coordinator
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    candidate = _candidate(source)
    candidate_hash = _candidate_hash(candidate)
    grant = await coordinator.reserve_grant(
        checkpoint.checkpoint_id,
        candidate_hash=candidate_hash,
        source_keys=_candidate_source_keys(candidate),
        verified_source_keys=_verified_source_keys(
            admission_runtime.wiki,
            candidate,
            frozenset[str](),
            coordinator.provenance_session_id,
        ),
    )
    assert grant is not None

    await coordinator.begin_turn("a later prompt", "a later prompt")
    await coordinator.finish_grant(
        checkpoint.checkpoint_id, outcome="persisted", candidate_hash=candidate_hash
    )

    # The authority is gone and the checkpoint is retired, not left in limbo.
    assert coordinator.unconsumed_grant_count == 0
    assert coordinator.unresolved_count == 0


@pytest.mark.asyncio
async def test_a_candidate_with_no_sources_is_grounded_from_the_checkpoint(
    admission_runtime,
) -> None:
    """The scenario that made the tool unusable in practice.

    The checkpoint block names sources by path only. To restate them the model
    would have to guess the workspace id and reproduce a content hash byte for
    byte — it cannot, so it gave up and discarded every checkpoint. Since a
    model-supplied source never carried authority anyway, the runtime fills in
    what it observed.
    """
    checkpoint, source, _ = await _file_checkpoint(admission_runtime)
    page = WikiPage(
        logical_path="concepts/release-rule.md",
        title="Release rule",
        created=_NOW,
        updated=_NOW,
        tags=["release"],
        sources=[],  # the model supplies content, not provenance
        revision=1,
        body="Signed tags only for every release.\n",
    )
    bare = WikiCandidate(
        summary="Record the signed-tag release rule",
        pages=[PageChange(page=page, expected_revision=None)],
        sources=[],
        value="high",
    )
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=bare,
        )
    )

    assert not result.is_error, result.message
    assert _revision(admission_runtime) != before
    assert admission_runtime.wiki_coordinator.unconsumed_grant_count == 0
    # What got written is the runtime's own provenance, not anything invented.
    assert source.path is not None


@pytest.mark.asyncio
async def test_filling_never_widens_what_the_model_actually_claimed(
    admission_runtime,
) -> None:
    """A candidate that names its own sources is still checked in full."""
    checkpoint, _, _ = await _file_checkpoint(admission_runtime)
    unread = Path(str(admission_runtime.session.work_dir)) / "not-observed.md"
    unread.write_text("never opened", encoding="utf-8")
    invented = admission_runtime.wiki.registry.relative_source(
        admission_runtime.workspace_id, unread
    )
    before = _revision(admission_runtime)

    result = await Wiki(admission_runtime)(
        Params(
            operation="remember",
            checkpoint_id=checkpoint.checkpoint_id,
            candidate=_candidate(invented),
        )
    )

    assert result.is_error
    assert _revision(admission_runtime) == before


@pytest.mark.asyncio
async def test_nothing_is_filled_without_an_open_checkpoint(admission_runtime) -> None:
    coordinator = admission_runtime.wiki_coordinator
    checkpoint, _, _ = await _file_checkpoint(admission_runtime)
    await coordinator.discard(checkpoint.checkpoint_id, "not_useful")

    assert await coordinator.checkpoint_sources(checkpoint.checkpoint_id) == ()
    assert await coordinator.checkpoint_sources("invented") == ()
