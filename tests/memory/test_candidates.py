"""Facts noticed automatically, and the approval they still have to pass.

Persistent memory used to contain only what the agent thought to record at the
moment it came up. Extraction closes that gap; the queue is what keeps it from
also removing the user from the decision.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import kimi_cli.memory.archivist as archivist_module
from kimi_cli.memory.archivist import _parse_candidates
from kimi_cli.memory.candidates import (
    CANDIDATE_TTL_SECONDS,
    MAX_CANDIDATES,
    CandidateFile,
    MemoryCandidate,
)


def _file(tmp_path: Path) -> CandidateFile:
    return CandidateFile(tmp_path / "candidates.jsonl")


class TestParsing:
    def test_it_reads_an_array_out_of_a_chatty_reply(self) -> None:
        raw = 'Sure! [{"kind": "project", "content": "acls is at /x", "key": "acls/repo"}] done'
        got = _parse_candidates(raw, session_id="s1")
        assert [(c.kind, c.content, c.key) for c in got] == [
            ("project", "acls is at /x", "acls/repo")
        ]

    def test_a_bad_row_is_dropped_without_taking_the_others(self) -> None:
        raw = """[
            {"kind": "bogus", "content": "wrong kind"},
            {"content": "no kind at all"},
            {"kind": "feedback", "content": "always check the date"},
            {"kind": "project", "content": "   "}
        ]"""
        assert [c.content for c in _parse_candidates(raw, session_id="s")] == [
            "always check the date"
        ]

    def test_an_unusable_key_loses_the_key_not_the_fact(self) -> None:
        """A key that only fails on promotion turns an approval into an error."""
        raw = '[{"kind": "project", "content": "a fact", "key": "not a valid key!"}]'
        got = _parse_candidates(raw, session_id="s")
        assert len(got) == 1
        assert got[0].key is None

    def test_nothing_qualifying_is_the_common_answer(self) -> None:
        assert _parse_candidates("[]", session_id="s") == []
        assert _parse_candidates("I could not find anything.", session_id="s") == []
        assert _parse_candidates("", session_id="s") == []

    def test_a_flood_of_proposals_is_capped(self) -> None:
        raw = (
            "[" + ",".join(f'{{"kind": "project", "content": "fact {i}"}}' for i in range(50)) + "]"
        )
        assert len(_parse_candidates(raw, session_id="s")) <= 5


class TestQueue:
    def test_a_restatement_of_something_queued_is_not_queued_twice(self, tmp_path: Path) -> None:
        queue = _file(tmp_path)
        queue.add([MemoryCandidate(kind="project", content="acls is at /x")])
        queue.add([MemoryCandidate(kind="project", content="  ACLS IS AT /X  ")])
        assert len(queue.read()) == 1

    def test_the_queue_stays_small(self, tmp_path: Path) -> None:
        """A backlog nobody clears is noise in every future session."""
        queue = _file(tmp_path)
        queue.add([MemoryCandidate(kind="project", content=f"fact {i}") for i in range(40)])
        assert len(queue.read()) == MAX_CANDIDATES

    def test_a_proposal_nobody_acted_on_expires(self, tmp_path: Path) -> None:
        queue = _file(tmp_path)
        stale = MemoryCandidate(
            kind="project", content="old news", created_at=time.time() - CANDIDATE_TTL_SECONDS - 1
        )
        fresh = MemoryCandidate(kind="project", content="current")
        queue.write([stale, fresh])
        assert [c.content for c in queue.read()] == ["current"]

    def test_taking_one_leaves_the_rest(self, tmp_path: Path) -> None:
        queue = _file(tmp_path)
        queue.add(
            [
                MemoryCandidate(kind="project", content="a"),
                MemoryCandidate(kind="project", content="b"),
            ]
        )
        first = queue.read()[0]
        assert queue.take(first.id).content == "a"
        assert [c.content for c in queue.read()] == ["b"]
        assert queue.take(first.id) is None

    def test_a_corrupt_line_does_not_take_the_file_with_it(self, tmp_path: Path) -> None:
        queue = _file(tmp_path)
        queue.path.write_text('{"broken\nnot json at all\n', encoding="utf-8")
        assert queue.read() == []

    def test_a_missing_file_is_simply_empty(self, tmp_path: Path) -> None:
        assert _file(tmp_path).read() == []


class TestExtractionCall:
    """The call itself, because a wrong signature here fails silently.

    `propose_candidates` swallows every exception by design — a failed
    extraction must not break compaction — which means a broken call looks
    exactly like "nothing worth keeping".
    """

    async def test_it_queues_what_the_model_returns(self, runtime, monkeypatch, tmp_path) -> None:
        from kosong.message import Message, TextPart

        import kimi_cli.memory.archivist as archivist
        from kimi_cli.memory.candidates import CANDIDATES_FILENAME, CandidateFile

        async def fake_ask(soul, conversation):
            return '[{"kind": "project", "content": "acls is at /x", "key": "acls/repo"}]'

        monkeypatch.setattr(archivist, "_ask_for_candidates", fake_ask)

        class _Soul:
            def __init__(self, rt):
                self.runtime = rt

        history = [Message(role="user", content=[TextPart(text="x" * 400)])]
        n = await archivist.propose_candidates(_Soul(runtime), history)

        assert n == 1
        queued = CandidateFile(runtime.user_memory_dir / CANDIDATES_FILENAME).read()
        assert [c.content for c in queued] == ["acls is at /x"]

    async def test_a_short_conversation_is_not_worth_extracting_from(
        self, runtime, monkeypatch
    ) -> None:
        from kosong.message import Message, TextPart

        import kimi_cli.memory.archivist as archivist

        called = False

        async def fake_ask(soul, conversation):
            nonlocal called
            called = True
            return "[]"

        monkeypatch.setattr(archivist, "_ask_for_candidates", fake_ask)

        class _Soul:
            def __init__(self, rt):
                self.runtime = rt

        n = await archivist.propose_candidates(
            _Soul(runtime), [Message(role="user", content=[TextPart(text="hi")])]
        )
        assert n == 0
        assert called is False, "no LLM call for a conversation with nothing in it"

    async def test_a_failing_extraction_does_not_break_compaction(
        self, runtime, monkeypatch
    ) -> None:
        from kosong.message import Message, TextPart

        import kimi_cli.memory.archivist as archivist

        async def boom(soul, conversation):
            raise RuntimeError("provider is down")

        monkeypatch.setattr(archivist, "_ask_for_candidates", boom)

        class _Soul:
            def __init__(self, rt):
                self.runtime = rt

        assert (
            await archivist.propose_candidates(
                _Soul(runtime), [Message(role="user", content=[TextPart(text="x" * 400)])]
            )
            == 0
        )


class TestExtractionPromptShape:
    """The transcript must sit *inside* the prompt, not after it.

    With the conversation appended last, the model treats its final turn as the
    live one and continues it instead of analysing it — against real sessions
    it replied to the transcript, or emitted the tool call the transcript was
    about to make, and returned no JSON at all. These assertions are cheap and
    the failure they guard against is silent: `propose_candidates` swallows
    everything, so an unparseable answer is indistinguishable from "nothing
    worth keeping".
    """

    def test_the_conversation_is_embedded_and_the_task_stated_after_it(self) -> None:
        from kimi_cli.memory.archivist import _EXTRACTION_PROMPT

        prompt = _EXTRACTION_PROMPT.format(
            conversation="user: hello\nassistant: hi", today="2026-03-05"
        )

        assert "<transcript>\nuser: hello" in prompt
        assert prompt.index("</transcript>") < prompt.index("JSON array"), (
            "the instruction has to come after the transcript, or it is what gets continued"
        )
        assert not prompt.rstrip().endswith("hi")

    def test_the_call_builds_the_prompt_rather_than_concatenating(self, monkeypatch) -> None:
        """A stray `PROMPT + conversation` would pass every other test here."""
        from kimi_cli.memory.archivist import _EXTRACTION_PROMPT

        assert "{conversation}" in _EXTRACTION_PROMPT
        source = inspect.getsource(archivist_module._ask_for_candidates)
        assert "_EXTRACTION_PROMPT.format(" in source
        assert "conversation=conversation" in source


class TestTimeAnchoring:
    """A fact anchored to a moment has to say which moment.

    Measured on LoCoMo, where every question asks when something happened,
    extraction that produced only timeless traits scored 6.7% — but the more
    telling number came from real sessions: 28 facts extracted across six
    conversations, not one carrying a date. Entries do have a `created_at`,
    and it answers a different question — when the fact was *recorded*, not
    when it was *true*.
    """

    def test_the_prompt_supplies_todays_date(self) -> None:
        """Without it, resolving "last Tuesday" means inventing a date."""
        from kimi_cli.memory.archivist import _EXTRACTION_PROMPT

        assert "{today}" in _EXTRACTION_PROMPT

        prompt = _EXTRACTION_PROMPT.format(conversation="x", today="2026-03-05")

        assert "2026-03-05" in prompt

    def test_the_call_passes_a_real_date(self) -> None:
        source = inspect.getsource(archivist_module._ask_for_candidates)

        assert "today=" in source, "the placeholder is useless if nothing fills it"
        assert "time.strftime" in source

    def test_dating_does_not_reopen_the_door_to_the_work_log(self) -> None:
        """Asking for dates invites recording what happened, which is excluded.

        The prompt has to hold both at once, so both are asserted: the rule
        that a decision carries its date, and the rule that activity does not
        become a memory just because it can be dated.
        """
        from kimi_cli.memory.archivist import _EXTRACTION_PROMPT

        assert "what was done" in _EXTRACTION_PROMPT
        assert "not licence to record what happened" in _EXTRACTION_PROMPT
        assert "a wrong date is worse than none" in _EXTRACTION_PROMPT


class TestSilentFailureIsNotSilent:
    """ "Nothing worth keeping" and "the extractor is broken" are the same shape.

    Both end as zero candidates, and that is how a prompt that never once
    produced a usable proposal survived for the life of the feature: the
    failure looked exactly like the common, correct answer. The evidence that
    separates them is in the reply itself and costs nothing to keep.
    """

    def test_an_empty_array_is_a_refusal(self) -> None:
        from kimi_cli.memory.archivist import _looks_like_refusal

        assert _looks_like_refusal("[]")
        assert _looks_like_refusal("  [ ]  ")
        assert _looks_like_refusal(""), "no reply proposed nothing either"

    def test_anything_else_that_parsed_to_nothing_is_a_fault(self) -> None:
        from kimi_cli.memory.archivist import _looks_like_refusal

        # What the model actually did when the transcript came last: it
        # continued the conversation instead of analysing it.
        assert not _looks_like_refusal("Sure — I'll run the tests now.")
        assert not _looks_like_refusal('<tool_calls><invoke name="bash">')
        assert not _looks_like_refusal('[{"kind": "project", "content":')

    async def _warnings_from(self, runtime, monkeypatch, reply: str) -> list[str]:
        """Run extraction against *reply* and return what it warned about.

        The logger is loguru, so it is intercepted directly rather than through
        `caplog`, which only sees the standard library.
        """
        from kosong.message import Message, TextPart

        import kimi_cli.memory.archivist as archivist

        async def fake_ask(soul, conversation):
            return reply

        monkeypatch.setattr(archivist, "_ask_for_candidates", fake_ask)

        seen: list[str] = []
        monkeypatch.setattr(archivist.logger, "warning", lambda message, **kw: seen.append(message))

        class _Soul:
            def __init__(self, rt):
                self.runtime = rt

        await archivist.propose_candidates(
            _Soul(runtime), [Message(role="user", content=[TextPart(text="x" * 400)])]
        )
        return seen

    async def test_an_unreadable_reply_is_reported(self, runtime, monkeypatch) -> None:
        seen = await self._warnings_from(
            runtime, monkeypatch, "I'll go ahead and check the branch."
        )

        assert any("nothing usable" in m for m in seen), (
            "a broken extractor has to be distinguishable from a quiet one"
        )

    async def test_a_refusal_is_not_reported_as_a_fault(self, runtime, monkeypatch) -> None:
        """Warning on the common answer would train everyone to ignore the warning."""
        assert await self._warnings_from(runtime, monkeypatch, "[]") == []
