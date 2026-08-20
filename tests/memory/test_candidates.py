"""The seam between this application and the memory package.

What extraction proposes, how the prompt is shaped, and how a queue expires
are tested in Amem, where that code lives — carrying copies here would put the
same assertions in two places and let them disagree, which is how the copy of
the implementation drifted in the first place.

What is tested here is the part Amem cannot see: turning kosong ``Message``
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

        Amem is handed a string; turning a list of kosong messages into that
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
    """Amem must not learn about kosong, and this is where that could slip.

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
