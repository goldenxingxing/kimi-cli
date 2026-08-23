"""The only boundary through which Wiki trigger events may reach telemetry.

Every Wiki event carries an allowlist of field names. Anything not on that list
is dropped before the event leaves this module, so a future caller cannot leak a
prompt, a summary, a Shell command, a credential, or an absolute path into a
telemetry sink by adding one keyword argument.

Emitting is always best effort: a telemetry or logging failure must never change
retrieval, the value gate, Approval, or a commit.
"""

from __future__ import annotations

from typing import Literal

from kimi_cli.telemetry import track
from kimi_cli.utils.logging import logger

WikiEventName = Literal[
    "wiki_evidence_recorded",
    "wiki_checkpoint_created",
    "wiki_checkpoint_resolved",
    "wiki_candidate_discarded",
    "wiki_approval_requested",
    "wiki_committed",
    "wiki_trigger_failed",
]

SafeScalar = bool | int | float | str | None

CheckpointOutcome = Literal["persist", "discard", "unresolved", "cancelled", "unavailable"]

_EVENT_FIELDS: dict[WikiEventName, frozenset[str]] = {
    "wiki_evidence_recorded": frozenset(
        {"producer_role", "evidence_class", "reliable", "stable", "source_count", "triggering"}
    ),
    "wiki_checkpoint_created": frozenset(
        {"cause", "producer_role", "evidence_count", "checkpoint_id"}
    ),
    "wiki_checkpoint_resolved": frozenset({"outcome", "duration_ms", "checkpoint_id"}),
    "wiki_candidate_discarded": frozenset({"reason", "checkpoint_id", "page_count"}),
    "wiki_approval_requested": frozenset({"mode", "page_count", "checkpoint_id"}),
    "wiki_committed": frozenset({"page_count", "global_revision", "checkpoint_id"}),
    "wiki_trigger_failed": frozenset({"stage", "error_class"}),
}


def track_wiki_event(event: WikiEventName, **fields: SafeScalar) -> None:
    """Emit one Wiki event, keeping only its allowlisted fields."""
    allowed = _EVENT_FIELDS.get(event)
    if allowed is None:
        return
    safe = {key: value for key, value in fields.items() if key in allowed}
    try:
        track(event, **safe)
    except Exception:
        logger.debug("Wiki telemetry failed", exc_info=True)


def allowed_fields(event: WikiEventName) -> frozenset[str]:
    """Expose the allowlist so tests can assert it rather than restate it."""
    return _EVENT_FIELDS[event]
