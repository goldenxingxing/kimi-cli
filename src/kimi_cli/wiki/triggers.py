"""Runtime-local trusted evidence and checkpoint coordination for the Wiki."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Literal
from uuid import UUID, uuid4

from kimi_cli.telemetry import track
from kimi_cli.wiki.models import SourceRef, validate_relative_source_path
from kimi_cli.wiki.schema import content_hash

CheckpointCause = Literal["root_evidence", "subagent_result", "explicit_user_durable"]
CheckpointState = Literal["pending", "persisting", "discarded", "consumed", "cancelled"]
CheckpointDiscardReason = Literal["user_declined", "not_useful", "superseded", "cancelled"]
EvidenceClass = Literal[
    "workspace-file",
    "workspace-search",
    "shell-result",
    "web-search",
    "web-document",
    "workspace-mutation",
]
ProducerRole = Literal["root", "subagent"]

_MAX_UNRESOLVED_CHECKPOINTS = 16
_MAX_BATCH_CHECKPOINTS = 4
_MAX_CHECKPOINT_EVIDENCE = 8
_MAX_CHECKPOINT_SUMMARY_BYTES = 1024
_MAX_BATCH_BYTES = 6 * 1024
_PRODUCER_ROLES = frozenset({"root", "subagent"})
_CHECKPOINT_CAUSES = frozenset({"root_evidence", "subagent_result", "explicit_user_durable"})
_CHECKPOINT_DISCARD_REASONS = frozenset({"user_declined", "not_useful", "superseded", "cancelled"})
_EXPECTED_SOURCE_KIND: dict[EvidenceClass, Literal["workspace-file", "conversation", "web"]] = {
    "workspace-file": "workspace-file",
    "workspace-search": "workspace-file",
    "shell-result": "conversation",
    "web-search": "web",
    "web-document": "web",
    "workspace-mutation": "workspace-file",
}


class WikiTriggerRejected(RuntimeError):
    """Raised when untrusted or stale runtime state attempts a transition."""


class WikiCheckpointBackpressure(WikiTriggerRejected):
    """Raised instead of evicting a qualifying unresolved checkpoint."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RootTurn:
    root_turn_id: str
    raw_hash: str
    normalized_hash: str
    started_at: float


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceObservation:
    root_turn_id: str
    workspace_id: UUID | None = None
    producer_role: ProducerRole
    producer_id: str | None
    run_generation: int | None
    tool_call_id: str
    source_class: EvidenceClass
    request_hash: str
    result_hash: str
    logical_paths: tuple[str, ...]
    source_refs: tuple[SourceRef, ...]
    reliable: bool
    stable_snapshot: bool
    triggering: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class WikiEvidence:
    evidence_id: str
    root_turn_id: str
    session_provenance_id: UUID
    workspace_id: UUID | None
    producer_role: ProducerRole
    producer_id: str | None
    run_generation: int | None
    tool_call_id: str
    source_class: EvidenceClass
    request_hash: str
    result_hash: str
    logical_paths: tuple[str, ...]
    source_refs: tuple[SourceRef, ...]
    reliable: bool
    stable_snapshot: bool
    triggering: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class WikiCheckpoint:
    checkpoint_id: str
    root_turn_id: str
    cause: CheckpointCause
    evidence_ids: tuple[str, ...]
    summary_hash: str | None
    producer_id: str | None
    run_generation: int | None
    dedupe_key: str
    state: CheckpointState
    created_at: float


@dataclass(frozen=True, slots=True, kw_only=True)
class WikiAdmissionGrant:
    checkpoint_id: str
    root_turn_id: str
    session_provenance_id: UUID
    workspace_id: UUID | None
    allowed_source_keys: frozenset[str]
    candidate_hash: str
    high_value: bool
    stable: bool
    grounded: bool
    reliable_source: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointBatch:
    checkpoints: tuple[WikiCheckpoint, ...]
    rendered: str


def canonical_digest(parts: Sequence[str]) -> str:
    """Hash ordered UTF-8 parts without allowing boundary ambiguity."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def canonical_evidence_source_digest(evidence: WikiEvidence) -> str:
    """Hash one observation's source identity independently of its producer/request."""
    source_class = evidence.source_class.strip().casefold().replace("_", "-")
    logical_paths = sorted(evidence.logical_paths)
    source_keys = sorted(_source_key(source) for source in evidence.source_refs)
    return canonical_digest(
        (
            evidence.session_provenance_id.hex,
            _uuid_component(evidence.workspace_id),
            source_class,
            evidence.result_hash,
            str(len(logical_paths)),
            *logical_paths,
            str(len(source_keys)),
            *source_keys,
        )
    )


class WikiTurnCoordinator:
    """Own one root runtime's trusted turn/evidence/checkpoint state.

    The coordinator intentionally keeps all state process-local.  It persists
    neither user text nor Wiki content, and records only canonical hashes and
    source metadata needed for later admission checks.
    """

    def __init__(
        self,
        *,
        provenance_session_id: UUID,
        workspace_id: UUID | None = None,
        telemetry_track: Callable[..., None] = track,
    ) -> None:
        self.provenance_session_id = provenance_session_id
        self.workspace_id = workspace_id
        self._track = telemetry_track
        self._lock = asyncio.Lock()
        self._owning_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._active_root_turn_id: str | None = None
        self._turns: dict[str, RootTurn] = {}
        self._evidence: dict[str, WikiEvidence] = {}
        self._evidence_by_key: dict[str, str] = {}
        self._checkpoints: dict[str, WikiCheckpoint] = {}
        self._checkpoint_by_key: dict[str, str] = {}
        self._grants: dict[str, WikiAdmissionGrant] = {}
        self._last_retrieval_outcome: str | None = None
        self._retrieval_refs: dict[str, tuple[tuple[str, int, str], ...]] = {}

    @property
    def unresolved_count(self) -> int:
        return sum(
            checkpoint.state in {"pending", "persisting"}
            for checkpoint in self._checkpoints.values()
        )

    @property
    def active_turn_id(self) -> str | None:
        """Return the current root turn identifier without exposing mutable state."""
        return self._active_root_turn_id

    @property
    def last_retrieval_outcome(self) -> str | None:
        """Return the last safe retrieval outcome category for this runtime."""
        return self._last_retrieval_outcome

    async def record_retrieval_outcome(
        self,
        outcome: str,
        *,
        result_refs: tuple[tuple[str, int, str], ...] = (),
    ) -> None:
        """Retain only bounded retrieval identifiers, never query or page content."""
        async with self._locked():
            self._require_open()
            if not outcome or len(outcome.encode("utf-8")) > 64:
                raise WikiTriggerRejected("invalid retrieval outcome")
            for path, revision, result_hash in result_refs:
                try:
                    validate_relative_source_path(path)
                except ValueError as exc:
                    raise WikiTriggerRejected("unsafe retrieval path") from exc
                if type(revision) is not int or revision < 0 or not _is_sha256(result_hash):
                    raise WikiTriggerRejected("invalid retrieval result reference")
            self._last_retrieval_outcome = outcome
            if result_refs:
                root_turn_id = self._require_active_turn()
                self._retrieval_refs[root_turn_id] = result_refs

    async def begin_turn(self, raw_text: str, normalized_text: str) -> RootTurn:
        async with self._locked():
            self._require_open()
            turn = RootTurn(
                root_turn_id=uuid4().hex,
                raw_hash=content_hash(raw_text.encode("utf-8")),
                normalized_hash=content_hash(normalized_text.encode("utf-8")),
                started_at=time.monotonic(),
            )
            self._turns[turn.root_turn_id] = turn
            self._active_root_turn_id = turn.root_turn_id
            self._safe_track("wiki_trigger_turn_started")
            return turn

    async def record_evidence(self, observation: EvidenceObservation) -> WikiEvidence | None:
        async with self._locked():
            self._require_open()
            logical_paths, source_refs = self._validate_observation(observation)
            key = canonical_digest(
                (
                    self.provenance_session_id.hex,
                    _uuid_component(self.workspace_id),
                    observation.root_turn_id,
                    observation.producer_role,
                    observation.producer_id or "",
                    (
                        str(observation.run_generation)
                        if observation.run_generation is not None
                        else ""
                    ),
                    observation.tool_call_id,
                    observation.source_class,
                    observation.request_hash,
                    observation.result_hash,
                    *logical_paths,
                    *(_source_key(source) for source in source_refs),
                    str(observation.reliable),
                    str(observation.stable_snapshot),
                    str(observation.triggering),
                )
            )
            existing_id = self._evidence_by_key.get(key)
            if existing_id is not None:
                return _evidence_snapshot(self._evidence[existing_id])

            evidence = WikiEvidence(
                evidence_id=uuid4().hex,
                root_turn_id=observation.root_turn_id,
                session_provenance_id=self.provenance_session_id,
                workspace_id=self.workspace_id,
                producer_role=observation.producer_role,
                producer_id=observation.producer_id,
                run_generation=observation.run_generation,
                tool_call_id=observation.tool_call_id,
                source_class=observation.source_class,
                request_hash=observation.request_hash,
                result_hash=observation.result_hash,
                logical_paths=logical_paths,
                source_refs=source_refs,
                reliable=observation.reliable,
                stable_snapshot=observation.stable_snapshot,
                triggering=observation.triggering,
            )
            self._evidence[evidence.evidence_id] = evidence
            self._evidence_by_key[key] = evidence.evidence_id
            self._safe_track(
                "wiki_trigger_evidence_recorded", source_class=observation.source_class
            )
            return _evidence_snapshot(evidence)

    async def import_evidence(self, evidence: WikiEvidence) -> WikiEvidence:
        """Reject records from every other coordinator, including same-session ones."""
        async with self._locked():
            self._require_open()
            local = self._evidence.get(evidence.evidence_id)
            if local != evidence or evidence.session_provenance_id != self.provenance_session_id:
                raise WikiTriggerRejected("evidence belongs to a different runtime")
            assert local is not None
            return _evidence_snapshot(local)

    async def create_checkpoint(
        self,
        cause: CheckpointCause,
        *,
        evidence_ids: tuple[str, ...] = (),
        summary_hash: str | None = None,
        producer_id: str | None = None,
        run_generation: int | None = None,
    ) -> WikiCheckpoint:
        async with self._locked():
            self._require_open()
            if type(cause) is not str or cause not in _CHECKPOINT_CAUSES:
                raise WikiTriggerRejected("unknown checkpoint cause")
            root_turn_id = self._require_active_turn()
            self._validate_checkpoint_inputs(root_turn_id, evidence_ids, summary_hash)
            dedupe_key = self._checkpoint_dedupe_key(
                root_turn_id=root_turn_id,
                cause=cause,
                evidence_ids=evidence_ids,
                summary_hash=summary_hash,
                producer_id=producer_id,
                run_generation=run_generation,
            )
            existing_id = self._checkpoint_by_key.get(dedupe_key)
            if existing_id is not None:
                return self._checkpoints[existing_id]
            if self.unresolved_count >= _MAX_UNRESOLVED_CHECKPOINTS:
                self._safe_track("wiki_trigger_checkpoint_backpressure")
                raise WikiCheckpointBackpressure("runtime checkpoint capacity reached")

            checkpoint = WikiCheckpoint(
                checkpoint_id=uuid4().hex,
                root_turn_id=root_turn_id,
                cause=cause,
                evidence_ids=evidence_ids,
                summary_hash=summary_hash,
                producer_id=producer_id,
                run_generation=run_generation,
                dedupe_key=dedupe_key,
                state="pending",
                created_at=time.monotonic(),
            )
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            self._checkpoint_by_key[dedupe_key] = checkpoint.checkpoint_id
            self._safe_track("wiki_trigger_checkpoint_created", cause=cause)
            return checkpoint

    async def attach_root_evidence_to_equivalent_subagent(
        self,
        *,
        summary_hash: str,
        evidence_ids: tuple[str, ...],
    ) -> WikiCheckpoint | None:
        """Atomically merge root evidence into one unambiguous subagent checkpoint."""
        async with self._locked():
            self._require_open()
            root_turn_id = self._require_active_turn()
            self._validate_checkpoint_inputs(root_turn_id, evidence_ids, summary_hash)
            root_evidence = tuple(self._evidence[evidence_id] for evidence_id in evidence_ids)
            if not root_evidence or any(
                evidence.producer_role != "root" for evidence in root_evidence
            ):
                return None
            root_source_digests = {
                canonical_evidence_source_digest(evidence) for evidence in root_evidence
            }
            matches = [
                checkpoint
                for checkpoint in self._checkpoints.values()
                if checkpoint.root_turn_id == root_turn_id
                and checkpoint.cause == "subagent_result"
                and checkpoint.summary_hash == summary_hash
                and checkpoint.state in {"pending", "persisting"}
                and checkpoint.producer_id
                and type(checkpoint.run_generation) is int
                and checkpoint.run_generation >= 0
                and root_source_digests
                & {
                    canonical_evidence_source_digest(evidence)
                    for evidence_id in checkpoint.evidence_ids
                    if (evidence := self._evidence[evidence_id]).producer_role == "subagent"
                    and evidence.producer_id == checkpoint.producer_id
                    and evidence.run_generation == checkpoint.run_generation
                }
            ]
            semantic_keys = {
                (
                    self.provenance_session_id,
                    self.workspace_id,
                    checkpoint.root_turn_id,
                    checkpoint.cause,
                    checkpoint.summary_hash,
                    checkpoint.producer_id,
                    checkpoint.run_generation,
                )
                for checkpoint in matches
            }
            if len(matches) != 1 or len(semantic_keys) != 1:
                return None
            checkpoint = matches[0]
            combined = tuple(dict.fromkeys((*checkpoint.evidence_ids, *evidence_ids)))
            if len(combined) > _MAX_CHECKPOINT_EVIDENCE:
                return None
            if combined == checkpoint.evidence_ids:
                return checkpoint
            dedupe_key = self._checkpoint_dedupe_key(
                root_turn_id=checkpoint.root_turn_id,
                cause=checkpoint.cause,
                evidence_ids=combined,
                summary_hash=checkpoint.summary_hash,
                producer_id=checkpoint.producer_id,
                run_generation=checkpoint.run_generation,
            )
            collision = self._checkpoint_by_key.get(dedupe_key)
            if collision is not None and collision != checkpoint.checkpoint_id:
                return None
            updated = replace(checkpoint, evidence_ids=combined, dedupe_key=dedupe_key)
            # Keep the old key as an alias so a replay of the pre-merge checkpoint
            # returns the immutable replacement instead of creating a duplicate.
            self._checkpoint_by_key[dedupe_key] = checkpoint.checkpoint_id
            self._checkpoints[checkpoint.checkpoint_id] = updated
            self._safe_track("wiki_trigger_checkpoint_evidence_attached")
            return updated

    async def pending_batch(self) -> CheckpointBatch:
        async with self._locked():
            pending = [
                checkpoint
                for checkpoint in self._checkpoints.values()
                if checkpoint.state == "pending"
            ]
            pending.sort(
                key=lambda checkpoint: (
                    checkpoint.cause != "explicit_user_durable",
                    checkpoint.created_at,
                )
            )
            selected = tuple(pending[:_MAX_BATCH_CHECKPOINTS])
            return CheckpointBatch(checkpoints=selected, rendered=_render_batch(selected))

    async def discard(self, checkpoint_id: str, reason: CheckpointDiscardReason) -> None:
        async with self._locked():
            if type(reason) is not str or reason not in _CHECKPOINT_DISCARD_REASONS:
                raise WikiTriggerRejected("unknown checkpoint discard reason")
            checkpoint = self._checkpoint(checkpoint_id)
            if checkpoint.state in {"discarded", "consumed", "cancelled"}:
                return
            self._checkpoints[checkpoint_id] = replace(checkpoint, state="discarded")
            self._grants.pop(checkpoint_id, None)
            self._safe_track("wiki_trigger_checkpoint_discarded", reason=reason)

    async def cancel_turn(self, root_turn_id: str) -> None:
        async with self._locked():
            if root_turn_id not in self._turns:
                raise WikiTriggerRejected("unknown root turn for this runtime")
            for checkpoint_id, checkpoint in tuple(self._checkpoints.items()):
                if checkpoint.root_turn_id == root_turn_id and checkpoint.state in {
                    "pending",
                    "persisting",
                }:
                    self._checkpoints[checkpoint_id] = replace(checkpoint, state="cancelled")
                    self._grants.pop(checkpoint_id, None)
            if self._active_root_turn_id == root_turn_id:
                self._active_root_turn_id = None
            self._safe_track("wiki_trigger_turn_cancelled")

    async def close(self) -> None:
        async with self._locked():
            if self._closed:
                return
            for checkpoint_id, checkpoint in tuple(self._checkpoints.items()):
                if checkpoint.state in {"pending", "persisting"}:
                    self._checkpoints[checkpoint_id] = replace(checkpoint, state="cancelled")
            self._grants.clear()
            self._active_root_turn_id = None
            self._closed = True
            self._safe_track("wiki_trigger_runtime_closed")

    def _locked(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._owning_loop is None:
            self._owning_loop = loop
        elif self._owning_loop is not loop:
            raise WikiTriggerRejected("runtime state cannot cross event loops")
        return self._lock

    def _require_open(self) -> None:
        if self._closed:
            raise WikiTriggerRejected("runtime state is closed")

    def _require_active_turn(self) -> str:
        if self._active_root_turn_id is None:
            raise WikiTriggerRejected("no active root turn in this runtime")
        return self._active_root_turn_id

    def _validate_observation(
        self, observation: EvidenceObservation
    ) -> tuple[tuple[str, ...], tuple[SourceRef, ...]]:
        if observation.root_turn_id not in self._turns:
            raise WikiTriggerRejected("evidence root turn is not owned by this runtime")
        if observation.workspace_id is not None and observation.workspace_id != self.workspace_id:
            raise WikiTriggerRejected("evidence workspace does not match this runtime")
        if not _is_sha256(observation.request_hash) or not _is_sha256(observation.result_hash):
            raise WikiTriggerRejected("evidence hashes must be canonical SHA-256 values")
        if not observation.tool_call_id:
            raise WikiTriggerRejected("evidence requires a tool call identifier")
        if (
            type(observation.producer_role) is not str
            or observation.producer_role not in _PRODUCER_ROLES
        ):
            raise WikiTriggerRejected("unknown evidence producer role")
        if observation.run_generation is not None and (
            type(observation.run_generation) is not int or observation.run_generation < 0
        ):
            raise WikiTriggerRejected("invalid evidence run generation")
        if observation.producer_role == "root" and (
            observation.producer_id is not None or observation.run_generation is not None
        ):
            raise WikiTriggerRejected("root evidence cannot claim subagent identity")
        if observation.producer_role == "subagent" and (
            not observation.producer_id or observation.run_generation is None
        ):
            raise WikiTriggerRejected("subagent evidence requires a producer and run generation")

        try:
            logical_paths = tuple(
                validate_relative_source_path(path) for path in observation.logical_paths
            )
        except ValueError as exc:
            raise WikiTriggerRejected("unsafe logical path in evidence") from exc

        source_refs = tuple(_source_snapshot(source) for source in observation.source_refs)
        expected_kind = _EXPECTED_SOURCE_KIND.get(observation.source_class)
        if expected_kind is None:
            raise WikiTriggerRejected("unknown evidence source class")
        for source in source_refs:
            if source.kind != expected_kind:
                raise WikiTriggerRejected("source kind does not match evidence class")
            if source.kind == "workspace-file" and source.workspace_id != self.workspace_id:
                raise WikiTriggerRejected("source workspace does not match this runtime")
            if source.kind == "conversation" and source.session_id != self.provenance_session_id:
                raise WikiTriggerRejected("conversation source does not match this runtime session")
        return logical_paths, source_refs

    def _validate_checkpoint_inputs(
        self,
        root_turn_id: str,
        evidence_ids: tuple[str, ...],
        summary_hash: str | None,
    ) -> None:
        if len(evidence_ids) > _MAX_CHECKPOINT_EVIDENCE:
            raise WikiTriggerRejected("checkpoint evidence reference limit exceeded")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise WikiTriggerRejected("checkpoint evidence identifiers must be unique")
        if summary_hash is not None and (
            not _is_sha256(summary_hash)
            or len(summary_hash.encode("utf-8")) > _MAX_CHECKPOINT_SUMMARY_BYTES
        ):
            raise WikiTriggerRejected("checkpoint summary hash is invalid")
        for evidence_id in evidence_ids:
            evidence = self._evidence.get(evidence_id)
            if evidence is None or evidence.root_turn_id != root_turn_id:
                raise WikiTriggerRejected(
                    "checkpoint evidence is not owned by the active root turn"
                )

    def _checkpoint_dedupe_key(
        self,
        *,
        root_turn_id: str,
        cause: CheckpointCause,
        evidence_ids: tuple[str, ...],
        summary_hash: str | None,
        producer_id: str | None,
        run_generation: int | None,
    ) -> str:
        return canonical_digest(
            (
                self.provenance_session_id.hex,
                _uuid_component(self.workspace_id),
                root_turn_id,
                cause,
                *evidence_ids,
                summary_hash or "",
                producer_id or "",
                str(run_generation) if run_generation is not None else "",
            )
        )

    def _checkpoint(self, checkpoint_id: str) -> WikiCheckpoint:
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise WikiTriggerRejected("unknown checkpoint for this runtime")
        return checkpoint

    def _safe_track(self, event: str, **properties: str | int | float | bool | None) -> None:
        with suppress(Exception):
            self._track(event, **properties)


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _source_key(source: SourceRef) -> str:
    return source.model_dump_json(exclude_none=True)


def _source_snapshot(source: SourceRef) -> SourceRef:
    """Validate and copy caller-owned Pydantic source models before retaining them."""
    return SourceRef.model_validate(source.model_dump(mode="python"))


def _evidence_snapshot(evidence: WikiEvidence) -> WikiEvidence:
    """Return a record whose nested source models cannot mutate coordinator state."""
    return replace(
        evidence, source_refs=tuple(_source_snapshot(source) for source in evidence.source_refs)
    )


def _uuid_component(value: UUID | None) -> str:
    return value.hex if value is not None else ""


def _render_batch(checkpoints: tuple[WikiCheckpoint, ...]) -> str:
    lines = ["<OPENKIMO_WIKI_CHECKPOINTS>"]
    for checkpoint in checkpoints:
        evidence_ids = ",".join(checkpoint.evidence_ids)
        lines.append(
            "- "
            f"checkpoint_id={checkpoint.checkpoint_id} "
            f"cause={checkpoint.cause} "
            f"evidence_ids={evidence_ids} "
            f"summary_hash={checkpoint.summary_hash or ''}"
        )
    lines.append("</OPENKIMO_WIKI_CHECKPOINTS>")
    rendered = "\n".join(lines)
    if len(rendered.encode("utf-8")) > _MAX_BATCH_BYTES:
        raise WikiTriggerRejected("checkpoint batch rendering exceeded its byte budget")
    return rendered
