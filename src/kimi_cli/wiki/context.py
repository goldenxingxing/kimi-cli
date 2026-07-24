"""Bounded global Wiki awareness for agent system prompts."""

from __future__ import annotations

import re
from collections.abc import Sequence

TRUNCATION_MARKER = "<!-- Wiki index truncated -->"
WIKI_BLOCK_START = "<!-- OPENKIMO_GLOBAL_WIKI_START -->"
WIKI_BLOCK_END = "<!-- OPENKIMO_GLOBAL_WIKI_END -->"
WIKI_PROMPT_MAX_BYTES = 8 * 1024
WIKI_GUIDANCE = (
    "The global Wiki is shared across all workspaces.\n"
    "Use Wiki search/read for durable knowledge.\n"
    "Propose only durable, sourced conclusions for writing."
)
_BLOCK_PATTERN = re.compile(
    re.escape(WIKI_BLOCK_START) + r".*?" + re.escape(WIKI_BLOCK_END),
    flags=re.DOTALL,
)
_LEGACY_INDEX_LINE = (
    r"(?:## (?:Entities|Concepts|Comparisons|Sources|Queries|Lint)|"
    r"[ \t]*[-*+][ \t]+[^\n]*|" + re.escape(TRUNCATION_MARKER) + r"|[ \t]*)(?=\n|$)"
)
_LEGACY_BLOCK_PATTERN = re.compile(
    r"^# Global Wiki\n\n"
    + re.escape(WIKI_GUIDANCE)
    + r"(?:\n\n# Wiki Index(?:\n"
    + _LEGACY_INDEX_LINE
    + r")*)?",
    flags=re.MULTILINE,
)


def render_compact_index(
    index_text: str,
    *,
    max_bytes: int = 8192,
    max_entries: int = 80,
    hints: Sequence[str] = (),
) -> str:
    """Return a UTF-8-safe, entry-bounded view of a Wiki index.

    Small indexes are preserved verbatim apart from surrounding whitespace.
    When either limit is exceeded, only whole Markdown list entries are kept.
    Entries matching a hint are considered first, while ties preserve document
    order. Space for the truncation marker is always reserved.
    """
    marker_bytes = len(TRUNCATION_MARKER.encode("utf-8"))
    byte_limit = _validated_limit(max_bytes, minimum=marker_bytes, name="max_bytes")
    entry_limit = _validated_limit(max_entries, minimum=0, name="max_entries")

    normalized = index_text.strip()
    entries = [
        line.strip()
        for line in normalized.splitlines()
        if line.lstrip().startswith(("- ", "* ", "+ "))
    ]
    if len(entries) <= entry_limit and len(normalized.encode("utf-8")) <= byte_limit:
        return normalized

    normalized_hints = tuple(hint.strip().casefold() for hint in hints if hint.strip())
    ranked = sorted(
        enumerate(entries),
        key=lambda item: (
            not any(hint in item[1].casefold() for hint in normalized_hints),
            item[0],
        ),
    )

    title = next(
        (line.strip() for line in normalized.splitlines() if line.strip().startswith("# ")),
        "",
    )
    selected: list[str] = [title] if title else []
    for _, entry in ranked[:entry_limit]:
        candidate = "\n".join((*selected, entry, TRUNCATION_MARKER))
        if len(candidate.encode("utf-8")) <= byte_limit:
            selected.append(entry)

    rendered = "\n".join((*selected, TRUNCATION_MARKER))
    if len(rendered.encode("utf-8")) > byte_limit:
        return TRUNCATION_MARKER
    return rendered


def build_wiki_context(index_text: str, *, hints: Sequence[str] = ()) -> str:
    """Build the guidance and compact index within the total marked-block budget."""
    fixed = f"{WIKI_BLOCK_START}\n# Global Wiki\n{WIKI_GUIDANCE}\n\n\n{WIKI_BLOCK_END}"
    index_budget = WIKI_PROMPT_MAX_BYTES - len(fixed.encode("utf-8"))
    compact = render_compact_index(index_text, max_bytes=index_budget, hints=hints)
    return WIKI_GUIDANCE + (f"\n\n{compact}" if compact else "")


def refresh_wiki_prompt_block(system_prompt: str, wiki_context: str) -> str:
    """Replace/insert the one managed Wiki block while preserving all other prompt text."""
    matches = _wiki_block_spans(system_prompt)
    if wiki_context:
        block = _render_prompt_block(wiki_context)
        if matches:
            parts: list[str] = []
            cursor = 0
            for index, (start, end) in enumerate(matches):
                parts.append(system_prompt[cursor:start])
                if index == 0:
                    parts.append(block)
                cursor = end
            parts.append(system_prompt[cursor:])
            return "".join(parts)
        anchor = "\n# Skills"
        if anchor in system_prompt:
            before, after = system_prompt.split(anchor, 1)
            return f"{before}{block}{anchor}{after}"
        return f"{system_prompt}\n\n{block}" if system_prompt else block
    if not matches:
        return system_prompt
    parts = []
    cursor = 0
    for start, end in matches:
        parts.append(system_prompt[cursor:start])
        cursor = end
    parts.append(system_prompt[cursor:])
    return "".join(parts)


def _render_prompt_block(wiki_context: str) -> str:
    block = f"{WIKI_BLOCK_START}\n# Global Wiki\n{wiki_context}\n{WIKI_BLOCK_END}"
    if len(block.encode("utf-8")) > WIKI_PROMPT_MAX_BYTES:
        raise ValueError("Wiki prompt block exceeds its UTF-8 byte budget")
    return block


def _wiki_block_spans(system_prompt: str) -> list[tuple[int, int]]:
    """Find marked blocks plus exact previous-Task-9 unmarked blocks."""
    marked = [(match.start(), match.end()) for match in _BLOCK_PATTERN.finditer(system_prompt)]
    legacy: list[tuple[int, int]] = []
    for match in _LEGACY_BLOCK_PATTERN.finditer(system_prompt):
        if any(start <= match.start() < end for start, end in marked):
            continue
        end = match.end()
        while end > match.start() and system_prompt[end - 1] == "\n":
            end -= 1
        legacy.append((match.start(), end))
    return sorted((*marked, *legacy))


def _validated_limit(value: object, *, minimum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value
