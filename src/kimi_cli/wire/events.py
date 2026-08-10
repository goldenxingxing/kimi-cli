"""Helpers for walking wire event payloads.

Lives in :mod:`kimi_cli.wire` rather than under ``vis/`` so that non-HTTP
consumers (analytics aggregators, CLI reporting) can reuse it without
importing a FastAPI router module.
"""

from __future__ import annotations

from typing import Any

__all__ = ["collect_events", "collect_events_with_origin", "MAIN_ORIGIN"]

MAIN_ORIGIN = "main"
"""Origin label for events emitted by the top-level agent."""

_UNKNOWN_SUBAGENT = "subagent"


def collect_events_with_origin(
    msg_type: str,
    payload: dict[str, Any],
    out: list[tuple[str, str, dict[str, Any]]],
    origin: str = MAIN_ORIGIN,
) -> None:
    """Unwrap ``SubagentEvent`` recursively, keeping track of who emitted each event.

    Appends ``(origin, type, payload)`` where *origin* is ``"main"`` for
    top-level events or the ``subagent_type`` of the innermost enclosing
    subagent (``"explore"``, ``"coder"``, …).

    Subagent activity is mirrored into the parent session's wire file wrapped in
    ``SubagentEvent`` envelopes, which nest arbitrarily deep, and in practice the
    majority of tool traffic lives in there. Plain :func:`collect_events`
    discards the envelope's identity fields; this variant keeps them so callers
    can tell automated subagent usage apart from what a user drove directly.

    For nested subagents the *innermost* type wins — that is the agent that
    actually issued the call.
    """
    if msg_type == "SubagentEvent":
        inner: dict[str, Any] | None = payload.get("event")
        if isinstance(inner, dict):
            inner_type: str = inner.get("type", "")
            inner_payload: dict[str, Any] = inner.get("payload", {})
            if inner_type:
                subagent_type = payload.get("subagent_type")
                child = (
                    subagent_type
                    if isinstance(subagent_type, str) and subagent_type
                    else _UNKNOWN_SUBAGENT
                )
                collect_events_with_origin(inner_type, inner_payload, out, child)
    else:
        out.append((origin, msg_type, payload))


def collect_events(
    msg_type: str,
    payload: dict[str, Any],
    out: list[tuple[str, dict[str, Any]]],
) -> None:
    """Recursively unwrap SubagentEvent and collect (type, payload) pairs.

    Origin-agnostic view, kept for callers that do not care who emitted the
    event. See :func:`collect_events_with_origin` when attribution matters.
    """
    with_origin: list[tuple[str, str, dict[str, Any]]] = []
    collect_events_with_origin(msg_type, payload, with_origin)
    out.extend((ev_type, ev_payload) for _origin, ev_type, ev_payload in with_origin)
