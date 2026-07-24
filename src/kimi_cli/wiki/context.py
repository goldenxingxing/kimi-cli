"""Bounded global Wiki awareness for agent system prompts."""

from __future__ import annotations

from collections.abc import Sequence

TRUNCATION_MARKER = "<!-- Wiki index truncated -->"


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


def _validated_limit(value: object, *, minimum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value
