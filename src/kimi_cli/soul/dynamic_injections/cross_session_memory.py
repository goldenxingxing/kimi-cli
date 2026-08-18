from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from kosong.message import Message

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.recent import (
    RECENT_FILENAME,
    SessionSummary,
    read_recent_summaries,
)
from kimi_cli.memory.storage import read_entries
from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_INJECTION_TYPE = "cross_session_memory"
_PERSISTENT_FILENAME = "persistent.jsonl"

# How many recent summaries to surface to the LLM at startup.
_RECENT_INJECTION_LIMIT = 5

# Ceiling on the whole snapshot. Persistent memory has no cap of its own — it
# only ever grows — so without this the opening cost of every session rises for
# the life of the account. Behavioural entries are written first and are never
# dropped: losing "be careful about X" silently changes how the agent works,
# where losing a project fact only means it has to be looked up.
_SNAPSHOT_BUDGET_CHARS = 12_000


class CrossSessionMemoryInjectionProvider(DynamicInjectionProvider):
    """One-shot startup injection of the user's cross-session memory.

    Reads ``persistent.jsonl`` (Memory tool entries) and ``recent.jsonl``
    (archived past-session summaries) once on the first LLM step and caches
    the rendered injection. Subsequent steps return ``[]`` so we don't pay
    file I/O on every step or invalidate prompt cache mid-session.
    """

    def __init__(self) -> None:
        self._injected: bool = False

    def invalidate(self) -> None:
        """Force a re-read on the next ``get_injections`` call."""
        self._injected = False

    @override
    async def on_context_compacted(self) -> None:
        """Re-inject after compaction.

        The snapshot is an ordinary history message, so compaction collapses it
        into the compaction summary. Without this the one-shot guard keeps
        returning the cached list against a history that no longer literally
        contains it, and the agent spends the rest of a long session with no
        cross-session memory at all — which is also why it re-records facts it
        has already stored.

        Costs nothing in prompt cache: compaction has already invalidated the
        whole prefix by definition.
        """
        self.invalidate()

    async def get_injections(
        self,
        history: Sequence[Message],
        soul: KimiSoul,
    ) -> list[DynamicInjection]:
        # Nothing to return once it is in history. The caller appends whatever
        # comes back as a *new* user message on every step, so handing back a
        # cached copy re-injected the whole snapshot each time — tens of
        # thousands of tokens per step, duplicated verbatim.
        if self._injected:
            return []

        self._injected = True
        try:
            user_memory_dir = soul.runtime.user_memory_dir
            persistent = read_entries(user_memory_dir / _PERSISTENT_FILENAME)
            recent = read_recent_summaries(
                user_memory_dir / RECENT_FILENAME,
                limit=_RECENT_INJECTION_LIMIT,
            )
        except Exception:
            logger.warning("cross-session memory read failed", exc_info=True)
            return []

        rendered = _render(persistent, recent)
        if not rendered:
            return []

        return [DynamicInjection(type=_INJECTION_TYPE, content=rendered)]


def _render(
    persistent: Sequence[MemoryEntry],
    recent: Sequence[SessionSummary],
) -> str:
    sections: list[str] = []

    behavioural = [e for e in persistent if e.is_behavioural]
    lookup = [e for e in persistent if not e.is_behavioural]

    if behavioural:
        lines = [
            "## Persistent memory",
            "Stable facts/preferences you've recorded across sessions:",
            "",
        ]
        lines.extend(e.render() for e in behavioural)
        sections.append("\n".join(lines))

    if lookup:
        # Listed, not quoted. These are facts about particular projects; most
        # are irrelevant to any given conversation, and carrying all of them
        # into every one costs more than fetching the occasional right answer.
        lines = [
            "## Recorded facts (index)",
            (
                "Summaries only. Read one in full with "
                "`Memory(operation={\"op\": \"get\", \"handle\": \"<handle>\"})` "
                "when it looks relevant to the task at hand:"
            ),
            "",
        ]
        lines.extend(e.render_index() for e in lookup)
        sections.append("\n".join(lines))

    if recent:
        lines = [
            "## Recent session summaries",
            "Condensed records of recent past conversations (oldest first):",
            "",
        ]
        for s in recent:
            lines.append(s.render())
            lines.append("")
        sections.append("\n".join(lines).rstrip())

    if not sections:
        return ""

    header = (
        "Cross-session memory — a snapshot of what you knew at the start of "
        "this conversation. Trust the live conversation over this snapshot if "
        "they conflict."
    )
    return _fit_budget(header + "\n\n" + "\n\n".join(sections))


def _fit_budget(text: str, budget: int = _SNAPSHOT_BUDGET_CHARS) -> str:
    """Hold the snapshot to ``budget``, dropping from the end.

    Sections are ordered so that what goes first is what must survive:
    behavioural memory, then the fact index, then session recaps. Cutting from
    the end therefore gives up recaps before facts and facts before
    instructions — and says so, rather than leaving the model to believe it has
    been shown everything.
    """
    if len(text) <= budget:
        return text
    head = text[:budget]
    boundary = head.rfind("\n")
    if boundary > budget // 2:
        head = head[:boundary]
    return head.rstrip() + "\n\n… (snapshot truncated; older entries omitted)"
