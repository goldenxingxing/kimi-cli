"""Reduce a session summary to what a *different* session can use.

What gets archived is the compaction LLM's output, and compaction writes a
handover for the session it is compacting: exact file states, absolute tool
paths, per-directory counts. All of it is correct, and almost none of it means
anything a week later in another conversation — but it is carried into every
later session's opening context, at roughly nine thousand tokens a time.

The compaction output is XML-sectioned, so the choice can be made structurally
rather than by asking a model to summarise a summary. Structural has two
properties an LLM pass does not: it costs nothing on the hot path, and it
applies to summaries already on disk. The full text is never modified — this
runs on the way out, so widening the selection later re-exposes everything.
"""

from __future__ import annotations

import re

#: Sections worth carrying across sessions, in the order they are rendered.
#:
#: What was being done, what is still wrong, and what was decided — the three
#: things a later session cannot reconstruct for itself.
CROSS_SESSION_SECTIONS: tuple[str, ...] = (
    "current_focus",
    "active_issues",
    "important_context",
)

#: Deliberately dropped, and why:
#:
#: - ``code_state`` (the largest, ~38%): file-by-file state, stale the moment
#:   the branch moves.
#: - ``environment``: absolute paths and tool locations, re-derivable and
#:   often wrong on another machine.
#: - ``completed_tasks``: what finished in *that* session; the durable part of
#:   it belongs in persistent memory, not in a recap.
DROPPED_SECTIONS: tuple[str, ...] = ("code_state", "environment", "completed_tasks")

#: Backstop for summaries the section pass cannot reduce — an older format, a
#: degraded raw-tail archive, or a kept section that is itself enormous.
DEFAULT_SUMMARY_BUDGET = 1_800

_SECTION_RE = "<{tag}>(.*?)</{tag}>"


def condense_summary(text: str, *, budget: int = DEFAULT_SUMMARY_BUDGET) -> str:
    """Return the cross-session-relevant part of a compaction summary.

    Falls back to the original text when it carries no recognised sections —
    a raw-tail archive written while the summarizer was unavailable still says
    something, and silently returning nothing would be worse than being long.
    Either way the result is held to ``budget``.
    """
    text = (text or "").strip()
    if not text:
        return ""

    kept: list[str] = []
    for tag in CROSS_SESSION_SECTIONS:
        match = re.search(_SECTION_RE.format(tag=tag), text, re.S)
        if match is None:
            continue
        body = match.group(1).strip()
        if body:
            kept.append(f"<{tag}>\n{body}\n</{tag}>")

    reduced = "\n".join(kept) if kept else text
    return _truncate(reduced, budget)


def _truncate(text: str, budget: int) -> str:
    """Cut to ``budget``, preferring a line boundary, and say that it was cut.

    Truncating mid-sentence invites the model to complete the thought it was
    left holding, so the marker is not decoration.
    """
    if budget <= 0 or len(text) <= budget:
        return text
    head = text[:budget]
    boundary = head.rfind("\n")
    if boundary > budget // 2:
        head = head[:boundary]
    return head.rstrip() + "\n… (truncated)"
