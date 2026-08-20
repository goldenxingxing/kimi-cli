from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from amem import render
from kosong.message import Message

from kimi_cli.memory.candidates import CANDIDATES_FILENAME, CandidateFile
from kimi_cli.memory.recent import RECENT_FILENAME, read_recent_summaries
from kimi_cli.memory.storage import PERSISTENT_FILENAME, read_entries
from kimi_cli.soul.dynamic_injection import DynamicInjection, DynamicInjectionProvider
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul

_INJECTION_TYPE = "cross_session_memory"

# How many recent summaries to surface to the LLM at startup.
_RECENT_INJECTION_LIMIT = 5

# The per-section ceilings, the ordering, and the wording of every section live
# in :func:`amem.render`. They were a second copy here — identical line for
# line apart from the function's name — which is the same duplication that had
# already let a shared helper drift, in a file that happens not to sit under
# kimi_cli/memory and so survived the first pass.


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
            persistent = read_entries(user_memory_dir / PERSISTENT_FILENAME)
            pending = CandidateFile(user_memory_dir / CANDIDATES_FILENAME).read()
            recent = read_recent_summaries(
                user_memory_dir / RECENT_FILENAME,
                limit=_RECENT_INJECTION_LIMIT,
            )
        except Exception:
            logger.warning("cross-session memory read failed", exc_info=True)
            return []

        rendered = render(persistent, recent, pending)
        if not rendered:
            return []

        return [DynamicInjection(type=_INJECTION_TYPE, content=rendered)]
