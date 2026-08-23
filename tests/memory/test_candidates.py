"""The seam between this application and the memory package.

What extraction proposes, how the prompt is shaped, and how a queue expires
are tested in Carryover, where that code lives — carrying copies here would put the
same assertions in two places and let them disagree, which is how the copy of
the implementation drifted in the first place.

What is tested here is the part Carryover cannot see: turning kosong ``Message``
objects into text, handing over a completer built from this soul's own model,
and writing the result where this application keeps it.
"""

from __future__ import annotations

import inspect

import kimi_cli.memory.archivist as archivist
from kimi_cli.memory.candidates import CANDIDATES_FILENAME, CandidateFile


class _Soul:
    """Enough of a soul for the adapter: it only reaches for ``runtime``."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime


def _history(text: str):
    from kosong.message import Message, TextPart

    return [Message(role="user", content=[TextPart(text=text)])]


def _compaction_result(summary: str):
    """What compaction hands the archivist.

    The summary is not a field on the result — it is the text of the first
    synthesized message, which is what `summary_from_compaction_result` reads.
    """
    from kosong.message import Message, TextPart

    from kimi_cli.soul.compaction import CompactionResult

    return CompactionResult(
        messages=[Message(role="user", content=[TextPart(text=summary)])], usage=None
    )


class TestTheAdapter:
    """`propose_candidates` swallows every exception by design.

    A failed extraction must not break compaction or shutdown, which means a
    wrong signature, a dead provider and "nothing worth keeping" all arrive as
    zero. Each is asserted separately for that reason: the silence is
    deliberate, so it cannot also be the only signal that something is wrong.
    """

    async def test_it_queues_what_the_model_proposed(self, runtime, monkeypatch) -> None:
        def completer(soul):
            async def complete(system: str, user: str) -> str:
                return '[{"kind": "project", "content": "the repo is at /x", "key": "svc/repo"}]'

            return complete

        monkeypatch.setattr(archivist, "_completer", completer)

        n = await archivist.propose_candidates(_Soul(runtime), _history("x" * 400))

        assert n == 1
        queued = CandidateFile(runtime.user_memory_dir / CANDIDATES_FILENAME).read()
        assert [c.content for c in queued] == ["the repo is at /x"]

    async def test_a_short_conversation_costs_no_request(self, runtime, monkeypatch) -> None:
        called = False

        def completer(soul):
            async def complete(system: str, user: str) -> str:
                nonlocal called
                called = True
                return "[]"

            return complete

        monkeypatch.setattr(archivist, "_completer", completer)

        assert await archivist.propose_candidates(_Soul(runtime), _history("hi")) == 0
        assert called is False, "no LLM call for a conversation with nothing in it"

    async def test_a_failing_provider_does_not_break_compaction(self, runtime, monkeypatch) -> None:
        def completer(soul):
            async def complete(system: str, user: str) -> str:
                raise RuntimeError("provider is down")

            return complete

        monkeypatch.setattr(archivist, "_completer", completer)

        assert await archivist.propose_candidates(_Soul(runtime), _history("x" * 400)) == 0

    async def test_the_conversation_reaches_the_completer_as_text(
        self, runtime, monkeypatch
    ) -> None:
        """The one thing only this side can get wrong.

        Carryover is handed a string; turning a list of kosong messages into that
        string is this application's job, and passing the wrong thing would
        surface as an extractor that proposes nothing.
        """
        seen: list[str] = []

        def completer(soul):
            async def complete(system: str, user: str) -> str:
                seen.append(user)
                return "[]"

            return complete

        monkeypatch.setattr(archivist, "_completer", completer)

        # Long enough to clear the threshold below which extraction is not
        # worth a request — otherwise nothing is called and this passes for
        # the wrong reason.
        marker = "remember the Windows port is 8721"
        await archivist.propose_candidates(_Soul(runtime), _history(marker + " " + "x" * 400))

        assert seen, "the completer was never called"
        assert marker in seen[0]


class TestTheProviderStaysOnThisSide:
    """Carryover must not learn about kosong, and this is where that could slip.

    The package imports no model client and reads no environment — that is why
    it installs anywhere. The coupling lives in ``_completer``, and keeping it
    there is what makes swapping the provider an edit in one function.
    """

    def test_the_completer_is_where_the_provider_is_named(self) -> None:
        source = inspect.getsource(archivist._completer)

        assert "kosong" in source

    def test_nothing_else_in_the_module_reaches_for_a_provider(self) -> None:
        source = inspect.getsource(archivist)
        outside = source.replace(inspect.getsource(archivist._completer), "")

        assert "chat_provider" not in outside


class TestWhereWritingIsWiredIn:
    """The adapter above is tested; being called is not the same thing.

    A real store showed nine session summaries, every one of them triggered by
    compaction and none by a session ending, and one proposal queued across
    nine sessions. That is the shape these tests pin: which paths reach
    extraction, and which do not reach it at all.
    """

    async def test_compaction_queues_a_proposal_on_disk(self, runtime, monkeypatch) -> None:
        """The path that does work, end to end through the file."""

        def completer(_soul):
            async def complete(system: str, user: str) -> str:
                return (
                    '[{"kind": "feedback", "content": "Ask before emailing a client", "key": null}]'
                )

            return complete

        monkeypatch.setattr(archivist, "_completer", completer)
        soul = _Soul(runtime)
        soul.context = None  # not read on this path

        await archivist.archive_compaction(
            soul,
            _compaction_result("a summary of the conversation"),
            history_before=_history("x" * 400),
        )

        queued = CandidateFile(runtime.user_memory_dir / CANDIDATES_FILENAME).read()
        assert [c.content for c in queued] == ["Ask before emailing a client"]

    def test_a_session_ending_does_not_extract(self) -> None:
        """Pinning the behaviour, which its own docstring contradicts.

        `propose_candidates` says it "costs one extra call at compaction and
        session end". It is called from one place, and that place is
        compaction. A session that never compacts proposes nothing — the
        common case for short ones, which is most of them.

        Asserted rather than fixed because fixing it buys an LLM call on every
        session end, including the ones where nothing happened, and that is a
        decision about cost rather than correctness.
        """
        source = inspect.getsource(archivist.archive_on_session_end)

        assert "propose_candidates" not in source
        assert "propose_candidates" in inspect.getsource(archivist.archive_compaction)

    def test_the_worker_never_archives_at_all(self) -> None:
        """The desktop app runs the worker, and the worker's shutdown does not.

        `archive_on_session_end` is reached from the terminal CLI's `finally`
        and from nowhere else, so in the app a session that does not compact
        leaves neither a summary nor a proposal. It is why every summary in a
        real store came from compaction.

        This test fails the day someone wires it up, which is the point: the
        gap is recorded rather than remembered.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src" / "kimi_cli"
        callers = [
            path.relative_to(src).as_posix()
            for path in src.rglob("*.py")
            if "archive_on_session_end" in path.read_text(encoding="utf-8")
            and path.name != "archivist.py"
        ]

        assert callers == ["cli/__init__.py"]
