"""What survives from a session summary into the next session's context."""

from __future__ import annotations

from kimi_cli.memory.condense import (
    CROSS_SESSION_SECTIONS,
    DROPPED_SECTIONS,
    condense_summary,
)

_FULL = """<current_focus>
Writing the survey; 97 PDFs downloaded.
</current_focus>

<environment>
- root: /Users/someone/Documents/work/output/papers/raw/
- pdftotext: /opt/homebrew/bin/pdftotext
</environment>

<completed_tasks>
- downloaded 97 PDFs across 8 categories
</completed_tasks>

<active_issues>
- duplicates across categories 03/04/05
</active_issues>

<code_state>
- src/a.py: rewritten
- src/b.py: untouched
</code_state>

<important_context>
- user wants category 05 filled in before writing
</important_context>"""


def test_it_keeps_what_another_session_cannot_reconstruct() -> None:
    out = condense_summary(_FULL)

    assert "Writing the survey" in out
    assert "duplicates across categories" in out
    assert "category 05 filled in before writing" in out


def test_it_drops_what_only_meant_something_in_that_session() -> None:
    """Absolute paths and file-by-file state are the bulk and the least useful."""
    out = condense_summary(_FULL)

    assert "/opt/homebrew/bin/pdftotext" not in out
    assert "src/a.py" not in out
    for tag in DROPPED_SECTIONS:
        assert f"<{tag}>" not in out


def test_it_is_a_large_reduction_on_a_realistic_summary() -> None:
    out = condense_summary(_FULL)

    assert len(out) < len(_FULL) / 2


def test_an_unrecognised_summary_is_kept_rather_than_lost() -> None:
    """A degraded raw-tail archive has no sections but still says something.

    Returning nothing here would silently erase the only record of a session
    whose summarizer was unavailable.
    """
    raw = "we talked about the deployment and decided to roll back"

    assert condense_summary(raw) == raw


def test_a_kept_section_that_is_itself_enormous_is_still_bounded() -> None:
    huge = "<current_focus>\n" + ("x" * 5_000) + "\n</current_focus>"

    out = condense_summary(huge, budget=500)

    assert len(out) <= 500 + len("\n… (truncated)")
    assert out.endswith("… (truncated)"), "a silent cut invites the model to finish the thought"


def test_truncation_prefers_a_line_boundary() -> None:
    body = "\n".join(f"line {i} with some content" for i in range(50))
    out = condense_summary(f"<current_focus>\n{body}\n</current_focus>", budget=200)

    assert "… (truncated)" in out
    assert not out.split("… (truncated)")[0].rstrip().endswith("wit")


def test_empty_stays_empty() -> None:
    assert condense_summary("") == ""
    assert condense_summary("   \n ") == ""


def test_the_two_section_lists_do_not_overlap() -> None:
    """A section cannot be both carried and dropped."""
    assert not set(CROSS_SESSION_SECTIONS) & set(DROPPED_SECTIONS)
