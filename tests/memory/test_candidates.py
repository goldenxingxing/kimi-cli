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

        prompt = _EXTRACTION_PROMPT.format(conversation="user: hello\nassistant: hi")

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
        assert "_EXTRACTION_PROMPT.format(conversation=conversation)" in source
