"""Extract conversation summaries and persist them as cross-session memory.

Two entry points:

- ``archive_compaction(soul, compaction_result)`` — extract the summary text
  the LLM has just produced for context compaction. No extra LLM call.

- ``archive_on_session_end(soul)`` — last-resort summary at shutdown. Calls the
  shared ``SimpleCompaction`` once on the current context; on any failure falls
  back to a raw text tail.

Both write to ``{user_memory_dir}/recent.jsonl`` with file locking.

What stays here is what is bound to this application: turning kosong ``Message``
objects into text, the compaction and shutdown hooks, and where the files live.
The extraction itself — the prompt, the parser, the refusal/fault distinction —
is :func:`carryover.propose`, reached through a ``Completer`` built from this soul's
own model. Nothing about the provider is configured; it is passed in.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from carryover.extract import TAIL_CHARS, Completer
from carryover.extract import propose as carryover_propose
from kosong.message import Message

from kimi_cli.memory.candidates import CANDIDATES_FILENAME, CandidateFile
from kimi_cli.memory.recent import (
    RECENT_FILENAME,
    SessionSummary,
    SummaryTrigger,
    append_summary,
)
from kimi_cli.memory.storage import PERSISTENT_FILENAME, stamp_relevance
from kimi_cli.soul.compaction import CompactionResult, SimpleCompaction
from kimi_cli.utils.logging import logger
from kimi_cli.wire.types import TextPart

if TYPE_CHECKING:
    from kimi_cli.soul.kimisoul import KimiSoul


_MIN_HISTORY_FOR_SESSION_END_SUMMARY = 4
RAW_FALLBACK_TAIL_MESSAGES = 6
RAW_FALLBACK_MAX_CHARS = 4_000


def extract_text(messages: Sequence[Message]) -> str:
    """Concatenate text content from messages, skipping internal ``<system>`` markers."""
    out: list[str] = []
    for msg in messages:
        for part in msg.content:
            if not isinstance(part, TextPart):
                continue
            text = part.text
            stripped = text.strip()
            if stripped.startswith("<system>") or stripped.startswith("<system-reminder>"):
                continue
            out.append(text)
    return "\n".join(t for t in (s.strip() for s in out) if t)


def summary_from_compaction_result(result: CompactionResult) -> str:
    """Pull the summary text the compaction LLM produced.

    ``result.messages[0]`` is a synthesized user message whose first content
    part is the ``<system>...compacted...</system>`` marker followed by the
    actual summary text parts.
    """
    if not result.messages:
        return ""
    return extract_text([result.messages[0]])


def raw_tail_summary(history: Sequence[Message]) -> str:
    """Cheap fallback: return the last few messages as plain text."""
    tail = list(history[-RAW_FALLBACK_TAIL_MESSAGES:])
    text = extract_text(tail)
    if len(text) > RAW_FALLBACK_MAX_CHARS:
        text = text[-RAW_FALLBACK_MAX_CHARS:]
    return text


async def _archive(
    soul: KimiSoul,
    summary_text: str,
    trigger: SummaryTrigger,
    *,
    degraded: bool = False,
) -> None:
    """Write one summary.

    ``degraded`` marks a raw conversation tail written because the summarizer
    was unavailable. It travels as an argument rather than a field on the
    record because it describes how the text was produced, not what it says —
    and adding a field would change every ``recent.jsonl`` on disk. Without it
    the store cannot tell a degraded ``session_end`` from a real one, and a
    transcript dump would supersede a good summary of the same session.
    """
    summary_text = summary_text.strip()
    if not summary_text:
        logger.debug("archivist: empty summary, skipping ({t})", t=trigger)
        return

    user_memory_dir = soul.runtime.user_memory_dir
    recent_path = user_memory_dir / RECENT_FILENAME

    work_dir_str: str | None = None
    try:
        work_dir_str = str(soul.runtime.session.work_dir)
    except Exception:
        work_dir_str = None

    summary = SessionSummary(
        session_id=soul.runtime.session.id,
        trigger=trigger,
        summary=summary_text,
        work_dir=work_dir_str,
    )
    try:
        result = append_summary(
            recent_path,
            summary,
            policy="skip_if_session_present" if degraded else "supersede",
        )
        if result.stored is None:
            logger.debug(
                "archivist: {t} summary for session {s} added nothing, dropped",
                t=trigger,
                s=summary.session_id[:8],
            )
            return
        logger.debug(
            "archivist: wrote {t} summary for session {s} ({n} chars){sup}",
            t=trigger,
            s=summary.session_id[:8],
            n=len(summary_text),
            sup=f", superseded {result.superseded_id[:8]}" if result.superseded_id else "",
        )
    except Exception:
        logger.warning("archivist: failed to write summary", exc_info=True)


async def archive_compaction(
    soul: KimiSoul,
    compaction_result: CompactionResult,
    *,
    history_before: Sequence[Message] | None = None,
) -> None:
    """Archive the summary produced by the most recent context compaction.

    ``history_before`` is what was about to be collapsed. Compaction is the
    last moment that detail exists, so it is where candidates are extracted
    from; without it the caller's history is already the summary.
    """
    text = summary_from_compaction_result(compaction_result)
    if not text:
        # Compaction may have been a no-op (too few messages to summarize).
        return
    await _archive(soul, text, "compaction")
    # The history about to be collapsed is the last chance to notice anything
    # in it worth keeping; after this it exists only as a summary.
    if history_before and soul.runtime.config.memory.propose_candidates:
        await propose_candidates(soul, history_before)


async def archive_on_session_end(soul: KimiSoul) -> None:
    """Best-effort summary at shutdown.

    Tries to summarize the current ``soul.context.history`` via
    ``SimpleCompaction``; on any failure (LLM unavailable, timeout, etc.)
    falls back to a raw text tail of the most recent messages so the user
    still gets *something* in their cross-session memory.
    """
    history = list(soul.context.history)
    if len(history) < _MIN_HISTORY_FOR_SESSION_END_SUMMARY:
        return

    summary_text = ""
    llm = soul.runtime.llm
    if llm is not None:
        try:
            compactor = SimpleCompaction()
            result = await compactor.compact(history, llm)
            summary_text = summary_from_compaction_result(result)
        except Exception:
            logger.warning("archivist: session-end LLM summary failed", exc_info=True)

    degraded = False
    if not summary_text:
        summary_text = raw_tail_summary(history)
        degraded = True

    if summary_text:
        await _archive(soul, summary_text, "session_end", degraded=degraded)


async def propose_candidates(soul: KimiSoul, history: Sequence[Message]) -> int:
    """Notice facts worth keeping and queue them for approval.

    Returns how many proposals were added. Never raises and never writes to
    persistent memory: a candidate is a suggestion, and the user still approves
    each one. What this removes is the requirement that the agent *think* to
    record something at the moment it comes up.

    Runs where summaries are already produced, so it costs one extra call at
    compaction and session end rather than anything per-turn.
    """
    try:
        text = extract_text(list(history))
        _note_what_came_up(soul, text[-TAIL_CHARS:])
        proposals = await carryover_propose(_completer(soul), text, session_id=soul.runtime.session.id)
        if not proposals:
            # carryover.propose already distinguishes "nothing worth keeping" from
            # "the reply could not be read" and logs which one happened. That
            # distinction is the reason this feature was broken for its whole
            # life once: both showed up here as no candidates.
            return 0
        CandidateFile(soul.runtime.user_memory_dir / CANDIDATES_FILENAME).add(proposals)
        logger.info("queued {n} memory candidate(s) for approval", n=len(proposals))
        return len(proposals)
    except Exception:
        logger.warning("memory candidate extraction failed", exc_info=True)
        return 0


def _note_what_came_up(soul: KimiSoul, conversation: str) -> None:
    """Record which behavioural entries this conversation was about.

    Runs where the extraction call already runs, so it costs a pass over text
    already in hand and no request. Failure is ignored on purpose: this feeds a
    suggestion, and losing a stamp delays a question rather than breaking
    anything.
    """
    try:
        stamp_relevance(
            soul.runtime.user_memory_dir / PERSISTENT_FILENAME,
            conversation,
            now=time.time(),
        )
    except Exception:
        logger.debug("could not record topical relevance", exc_info=True)


def _completer(soul: KimiSoul) -> Completer:
    """This soul's model, in the shape :func:`carryover.propose` asks for.

    The whole coupling to a provider is these fifteen lines. carryover imports no
    client and reads no environment: it is handed something that takes a system
    prompt and a user prompt and returns text, which is all it needs and all
    this has to promise.
    """

    async def complete(system: str, user: str) -> str:
        import kosong
        from kosong.tooling.empty import EmptyToolset

        llm = soul.runtime.llm
        if llm is None:
            return ""
        result = await kosong.step(
            llm.chat_provider,
            system,
            EmptyToolset(),
            [Message(role="user", content=[TextPart(text=user)])],
        )
        return extract_text([result.message])

    return complete
