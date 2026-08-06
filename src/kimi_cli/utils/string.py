from __future__ import annotations

import random
import re
import string
import unicodedata

_NEWLINE_RE = re.compile(r"[\r\n]+")


def fold_text(raw: str) -> str:
    """Fold Unicode width, whitespace, and case without dropping any content.

    Exactly three axes, chosen so the folded string still stands for the same
    statement: fullwidth and compatibility forms collapse to their canonical
    equivalents, runs of whitespace become single spaces, and case is removed.
    Punctuation, diacritics, and word order survive untouched.

    Shared by the Wiki's intent hashing and by memory deduplication. It lives
    here rather than in either caller because ``kimi_cli.memory`` is on the CLI
    startup path and must not pull in the Wiki package to fold a string.
    """
    return " ".join(unicodedata.normalize("NFKC", raw).split()).casefold()


def shorten(text: str, *, width: int, placeholder: str = "…") -> str:
    """Shorten text to at most *width* characters.

    Normalises whitespace, then truncates — preferring a word boundary
    when one exists near the cut point, but falling back to a hard cut
    so that CJK text without spaces won't collapse to just the placeholder.
    """
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    cut = width - len(placeholder)
    if cut <= 0:
        return text[:width]
    space = text.rfind(" ", 0, cut + 1)
    if space > 0:
        cut = space
    return text[:cut].rstrip() + placeholder


def shorten_middle(text: str, width: int, remove_newline: bool = True) -> str:
    """Shorten the text by inserting ellipsis in the middle."""
    if len(text) <= width:
        return text
    if remove_newline:
        text = _NEWLINE_RE.sub(" ", text)
    return text[: width // 2] + "..." + text[-width // 2 :]


def random_string(length: int = 8) -> str:
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))
