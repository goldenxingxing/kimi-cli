"""Shared scaffolding for scanning session ``wire.jsonl`` files.

This is the reusable half of what :mod:`kimi_cli.vis.api.statistics` does inline:
walk every session's wire log, unwrap ``SubagentEvent`` envelopes, and hand the
caller a flat stream of ``(timestamp, event_type, payload)``.

Performance notes (these matter — a mature install has hundreds of sessions and
wire files are dominated by nested subagent traffic):

* ``line_filter`` runs on the *raw line* before ``json.loads``. A C-level
  substring test rejects the overwhelming majority of lines for a fraction of
  the cost of parsing them.
* We use bare ``json.loads`` rather than ``parse_wire_file_line``. The envelope
  payload is already ``dict[str, JsonType]``, so consumers do dict access
  either way, and full pydantic validation of every record is roughly an order
  of magnitude more expensive.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from kimi_cli.metadata import WorkDirMeta, load_metadata
from kimi_cli.wire.events import collect_events_with_origin

__all__ = ["iter_session_dirs", "iter_wire_events", "WIRE_FILENAME"]

WIRE_FILENAME = "wire.jsonl"


def iter_session_dirs() -> Iterator[tuple[WorkDirMeta, Path]]:
    """Yield ``(work_dir_meta, session_dir)`` for every session with a wire log.

    Covers every registered work directory and, for each, both the current
    (``<work_dir>/session-data``) and legacy (``~/.kimi/sessions/<md5>``) roots
    via :attr:`WorkDirMeta.readable_sessions_dirs`.

    Only depth-1 children of a session root are considered. Nested
    ``<session>/subagents/<name>/wire.jsonl`` files are deliberately skipped:
    their events are already mirrored into the parent wire as ``SubagentEvent``
    envelopes, so scanning them would double count.
    """
    seen: set[Path] = set()
    for wd in load_metadata().work_dirs:
        for root in wd.readable_sessions_dirs:
            try:
                children = sorted(root.iterdir())
            except OSError:
                continue
            for session_dir in children:
                if not session_dir.is_dir():
                    continue
                if not (session_dir / WIRE_FILENAME).is_file():
                    continue
                # The same physical directory can be reachable from two work
                # dir entries (e.g. a path registered twice); dedupe by resolved
                # identity so counts stay honest.
                try:
                    key = session_dir.resolve()
                except OSError:
                    key = session_dir
                if key in seen:
                    continue
                seen.add(key)
                yield wd, session_dir


def iter_wire_events(
    wire_path: Path,
    *,
    line_filter: Callable[[str], bool] | None = None,
) -> Iterator[tuple[float, str, dict[str, Any], str]]:
    """Yield ``(timestamp, event_type, payload, origin)`` from one wire log.

    ``SubagentEvent`` envelopes are recursively unwrapped, so nested subagent
    activity surfaces as its own inner event type. *origin* is ``"main"`` for
    top-level events or the emitting subagent's type — most tool traffic in a
    real session comes from subagents, and the two are worth telling apart.

    Malformed lines — including a torn final line from a session that is still
    being written — are skipped silently rather than aborting the scan.
    """
    try:
        handle = wire_path.open(encoding="utf-8")
    except OSError:
        return
    with handle as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line_filter is not None and not line_filter(line):
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            # The first line is a WireFileMetadata header, which has no
            # "message" key; the same guard covers any other stray object.
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            msg_type = message.get("type")
            if not isinstance(msg_type, str):
                continue
            payload = message.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            timestamp = record.get("timestamp")
            ts = float(timestamp) if isinstance(timestamp, (int, float)) else 0.0

            unwrapped: list[tuple[str, str, dict[str, Any]]] = []
            collect_events_with_origin(msg_type, payload, unwrapped)
            for origin, ev_type, ev_payload in unwrapped:
                yield ts, ev_type, ev_payload, origin
