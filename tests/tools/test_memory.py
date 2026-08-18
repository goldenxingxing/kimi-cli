from __future__ import annotations

import json
from typing import Any

import pytest
from kosong.tooling import DisplayBlock, ToolReturnValue
from pydantic import ValidationError

from kimi_cli.memory.storage import read_entries
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval, ApprovalRequestPolicy, ApprovalResult
from kimi_cli.tools.memory import Memory, Params, UpdateOp
from tests.conftest import tool_call_context


class RecordingApproval(Approval):
    """Records every request so a test can assert on the action and the prompt."""

    def __init__(self, *, approve: bool = True) -> None:
        super().__init__(yolo=False)
        self.approve = approve
        self.requests: list[tuple[str, str]] = []

    async def request(
        self,
        sender: str,
        action: str,
        description: str,
        display: list[DisplayBlock] | None = None,
        *,
        request_policy: ApprovalRequestPolicy = "default",
    ) -> ApprovalResult:
        self.requests.append((action, description))
        return ApprovalResult(self.approve)


@pytest.fixture
def memory_tool(runtime: Runtime) -> Memory:
    return Memory(runtime)


async def add(tool: Memory, content: str, *, kind: str = "user", scope: str = "persistent"):
    return await tool(
        Params.model_validate(
            {"operation": {"op": "add", "kind": kind, "scope": scope, "content": content}}
        )
    )


def payload(result: ToolReturnValue) -> dict[str, Any]:
    assert not result.is_error
    return json.loads(result.output or "{}")


def persistent_file(runtime: Runtime):
    return runtime.user_memory_dir / "persistent.jsonl"


async def test_add_reports_a_creation_and_keeps_the_original_keys(
    memory_tool: Memory, runtime: Runtime
) -> None:
    with tool_call_context("Memory"):
        body = payload(await add(memory_tool, "User prefers terse replies"))

    assert body["result"] == "created"
    # Anything already reading these three keys must keep working.
    assert set(body) >= {"id", "scope", "kind"}
    assert body["scope"] == "persistent"
    assert body["kind"] == "user"


async def test_restating_a_fact_merges_instead_of_adding(
    memory_tool: Memory, runtime: Runtime
) -> None:
    with tool_call_context("Memory"):
        first = payload(await add(memory_tool, "User prefers terse replies"))
        second = payload(await add(memory_tool, "User prefers terse replies."))

    assert second["result"] == "merged"
    assert second["merged_into"] == first["id"]
    assert second["replaced"] == "User prefers terse replies"
    assert len(read_entries(persistent_file(runtime))) == 1


async def test_a_merged_id_can_still_be_updated_and_deleted(
    memory_tool: Memory, runtime: Runtime
) -> None:
    with tool_call_context("Memory"):
        first = payload(await add(memory_tool, "User prefers terse replies"))
        await add(memory_tool, "User prefers terse replies.")

        updated = await memory_tool(
            Params.model_validate(
                {"operation": {"op": "update", "id": first["id"], "content": "Revised"}}
            )
        )
        assert not updated.is_error
        assert read_entries(persistent_file(runtime))[0].content == "Revised"

        deleted = await memory_tool(
            Params.model_validate({"operation": {"op": "delete", "id": first["id"]}})
        )
        assert not deleted.is_error
        assert read_entries(persistent_file(runtime)) == []


async def test_a_near_miss_is_reported_rather_than_merged(
    memory_tool: Memory, runtime: Runtime
) -> None:
    with tool_call_context("Memory"):
        first = payload(await add(memory_tool, "Pipeline bugs tracked in Linear project INGEST"))
        second = payload(await add(memory_tool, "Pipeline bugs tracked in Linear project EGRESS"))

    assert second["result"] == "created"
    duplicates = second["possible_duplicates"]
    assert [d["id"] for d in duplicates] == [first["id"]]
    assert duplicates[0]["content"] == "Pipeline bugs tracked in Linear project INGEST"
    assert len(read_entries(persistent_file(runtime))) == 2


async def test_a_merge_asks_under_its_own_action_and_shows_both_texts(runtime: Runtime) -> None:
    approval = RecordingApproval()
    runtime.approval = approval
    tool = Memory(runtime)

    with tool_call_context("Memory"):
        await add(tool, "User prefers terse replies")
        await add(tool, "User prefers terse replies.")

    create_action, create_description = approval.requests[0]
    merge_action, merge_description = approval.requests[1]

    # Separate action strings keep the two consents apart: allowing the agent
    # to create entries is not consent to overwrite existing ones.
    assert create_action == "memory.add"
    assert merge_action == "memory.merge"

    assert "Add persistent memory (user)" in create_description
    assert "Merge persistent memory (user)" in merge_description
    # The older text is about to be destroyed; this prompt is the only place a
    # human can stop it, so it has to show what is being replaced.
    assert "User prefers terse replies" in merge_description
    assert "User prefers terse replies." in merge_description
    assert "100%" in merge_description or "%" in merge_description


async def test_a_rejected_merge_leaves_the_file_untouched(runtime: Runtime) -> None:
    approval = RecordingApproval(approve=True)
    runtime.approval = approval
    tool = Memory(runtime)

    with tool_call_context("Memory"):
        await add(tool, "User prefers terse replies")
        before = persistent_file(runtime).read_bytes()

        approval.approve = False
        rejected = await add(tool, "User prefers terse replies.")

    assert rejected.is_error
    assert persistent_file(runtime).read_bytes() == before


async def test_declining_a_memory_does_not_end_the_turn(runtime: Runtime) -> None:
    """Remembering is a side errand the agent starts while doing something else.

    Declining it used to end the whole turn — the root agent stops on any
    rejection that carries no feedback — so the task in progress was abandoned
    unfinished, with the user having said nothing about it.
    """
    from kimi_cli.tools.utils import ToolRejectedError

    approval = RecordingApproval(approve=False)
    runtime.approval = approval
    tool = Memory(runtime)

    with tool_call_context("Memory"):
        rejected = await add(tool, "Something the user would rather not keep")

    assert isinstance(rejected, ToolRejectedError)
    assert rejected.stops_turn is False
    assert "continue with the task" in rejected.message.lower()


async def test_session_scope_is_untouched_by_dedup(memory_tool: Memory, runtime: Runtime) -> None:
    with tool_call_context("Memory"):
        await add(memory_tool, "A session note", scope="session")
        body = payload(await add(memory_tool, "A session note", scope="session"))

    assert body["result"] == "created"
    assert len(runtime.session.state.session_memory) == 2
    assert not persistent_file(runtime).exists()


async def test_update_does_not_deduplicate(memory_tool: Memory, runtime: Runtime) -> None:
    # An update targets an id the caller chose. Cascading a merge from it would
    # silently delete a second entry, so it deliberately does not.
    with tool_call_context("Memory"):
        first = payload(await add(memory_tool, "User prefers terse replies"))
        second = payload(await add(memory_tool, "Deploy target is fly.io"))

        await memory_tool(
            Params.model_validate(
                {
                    "operation": {
                        "op": "update",
                        "id": second["id"],
                        "content": "User prefers terse replies",
                    }
                }
            )
        )

    stored = read_entries(persistent_file(runtime))
    assert len(stored) == 2
    assert {e.id for e in stored} == {first["id"], second["id"]}


def test_memory_params_accept_json_encoded_operation_object() -> None:
    params = Params.model_validate(
        {"operation": '{"op":"update","id":"memory-1","content":"new value"}'}
    )

    assert params.operation == UpdateOp(id="memory-1", content="new value")


@pytest.mark.parametrize("operation", ['"update"', "[]", "{broken", "update"])
def test_memory_params_reject_non_object_operation_strings(operation: str) -> None:
    with pytest.raises(ValidationError):
        Params.model_validate({"operation": operation})


# --- Reading one entry back -------------------------------------------------
#
# Facts are listed in the opening context as one-line summaries rather than in
# full, so the agent needs a way to read the one it decides is relevant.


async def _add_keyed(tool: Memory, content: str, key: str, kind: str = "project"):
    return await tool(
        Params.model_validate(
            {
                "operation": {
                    "op": "add",
                    "kind": kind,
                    "scope": "persistent",
                    "content": content,
                    "key": key,
                }
            }
        )
    )


async def _get(tool: Memory, handle: str):
    return await tool(Params.model_validate({"operation": {"op": "get", "handle": handle}}))


async def test_get_reads_an_entry_by_its_key(memory_tool: Memory) -> None:
    with tool_call_context("Memory"):
        await _add_keyed(memory_tool, "acls lives at /Users/x/acls", "acls/repo-path")
        body = payload(await _get(memory_tool, "acls/repo-path"))

    assert body["content"] == "acls lives at /Users/x/acls"
    assert body["key"] == "acls/repo-path"


async def test_get_reads_an_entry_by_id_for_records_written_before_keys(
    memory_tool: Memory,
) -> None:
    """An entry with no key can still only be addressed by its id."""
    with tool_call_context("Memory"):
        created = payload(await add(memory_tool, "an older fact", kind="project"))
        body = payload(await _get(memory_tool, created["id"][:8]))

    assert body["content"] == "an older fact"
    assert body["key"] is None


async def test_get_is_case_insensitive_about_the_handle(memory_tool: Memory) -> None:
    with tool_call_context("Memory"):
        await _add_keyed(memory_tool, "a fact", "acls/repo-path")
        body = payload(await _get(memory_tool, "ACLS/Repo-Path"))

    assert body["content"] == "a fact"


async def test_get_says_so_when_the_handle_is_unknown(memory_tool: Memory) -> None:
    """A wrong handle must be visible, not silently read as "no memory"."""
    with tool_call_context("Memory"):
        result = await _get(memory_tool, "no/such-thing")

    assert result.is_error
    assert "no/such-thing" in str(result)


async def test_a_key_that_could_not_be_typed_back_is_refused_at_the_tool(
    memory_tool: Memory,
) -> None:
    with pytest.raises(ValidationError):
        Params.model_validate(
            {
                "operation": {
                    "op": "add",
                    "kind": "project",
                    "scope": "persistent",
                    "content": "x",
                    "key": "not a valid key",
                }
            }
        )


# --- Finding an entry you cannot name ---------------------------------------


async def _search(tool: Memory, query: str):
    return await tool(Params.model_validate({"operation": {"op": "search", "query": query}}))


async def test_search_finds_an_entry_by_its_content(memory_tool: Memory) -> None:
    with tool_call_context("Memory"):
        await _add_keyed(memory_tool, "CodeGraph does not track git branches", "acls/codegraph")
        body = payload(await _search(memory_tool, "CodeGraph"))

    assert [h["handle"] for h in body["hits"]] == ["acls/codegraph"]


async def test_search_works_on_chinese_shorter_than_a_trigram(memory_tool: Memory) -> None:
    """Two characters is a whole word in Chinese, and below what FTS5 can index."""
    with tool_call_context("Memory"):
        await _add_keyed(memory_tool, "126 邮箱已接入并验证可用", "mail/126")
        body = payload(await _search(memory_tool, "邮箱"))

    assert [h["handle"] for h in body["hits"]] == ["mail/126"]


async def test_search_on_chinese_above_the_trigram_floor(memory_tool: Memory) -> None:
    with tool_call_context("Memory"):
        await _add_keyed(memory_tool, "真实仓库路径为 /Users/x/acls，不是副本", "acls/repo")
        body = payload(await _search(memory_tool, "仓库路径"))

    assert [h["handle"] for h in body["hits"]] == ["acls/repo"]


async def test_a_query_full_of_fts_syntax_is_matched_literally(memory_tool: Memory) -> None:
    """Models write `core.py` and `a/b-c`; FTS5 reads those as operators."""
    with tool_call_context("Memory"):
        await _add_keyed(memory_tool, "the file core.py lives under src/", "proj/core")
        body = payload(await _search(memory_tool, "core.py"))

    assert [h["handle"] for h in body["hits"]] == ["proj/core"]


async def test_search_reports_no_matches_rather_than_failing(memory_tool: Memory) -> None:
    with tool_call_context("Memory"):
        body = payload(await _search(memory_tool, "nothing stored about this"))

    assert body["hits"] == []


# --- Suggestions the agent did not have to think of --------------------------


def _queue(runtime: Runtime):
    from kimi_cli.memory.candidates import CANDIDATES_FILENAME, CandidateFile

    return CandidateFile(runtime.user_memory_dir / CANDIDATES_FILENAME)


def _suggest(runtime: Runtime, content: str, kind: str = "project", key: str | None = None):
    from kimi_cli.memory.candidates import MemoryCandidate

    candidate = MemoryCandidate(kind=kind, content=content, key=key)  # type: ignore[arg-type]
    _queue(runtime).add([candidate])
    return _queue(runtime).read()[-1]


async def _op(tool: Memory, **operation):
    return await tool(Params.model_validate({"operation": operation}))


async def test_promoting_a_suggestion_stores_it(memory_tool: Memory, runtime: Runtime) -> None:
    candidate = _suggest(runtime, "acls lives at /Users/x/acls", key="acls/repo")

    with tool_call_context("Memory"):
        body = payload(await _op(memory_tool, op="promote", id=candidate.id))

    assert body["promoted"] is True
    stored = read_entries(persistent_file(runtime))
    assert [e.content for e in stored] == ["acls lives at /Users/x/acls"]
    assert _queue(runtime).read() == [], "a kept suggestion leaves the queue"


async def test_a_refused_suggestion_stays_queued(runtime: Runtime) -> None:
    """It was noticed automatically; a "no" now is not a "no" forever.

    Dropping it on refusal would silently discard something the user might
    want raised again later, with no record that it was ever proposed.
    """
    approval = RecordingApproval(approve=False)
    runtime.approval = approval
    tool = Memory(runtime)
    candidate = _suggest(runtime, "a fact the user declined for now")

    with tool_call_context("Memory"):
        result = await _op(tool, op="promote", id=candidate.id)

    assert result.is_error
    assert not persistent_file(runtime).exists() or read_entries(persistent_file(runtime)) == []
    assert [c.id for c in _queue(runtime).read()] == [candidate.id]


async def test_dismissing_drops_it_without_asking(memory_tool: Memory, runtime: Runtime) -> None:
    """Nothing was stored, so there is nothing to approve."""
    candidate = _suggest(runtime, "not worth keeping")

    with tool_call_context("Memory"):
        body = payload(await _op(memory_tool, op="dismiss", id=candidate.id))

    assert body["dismissed"] is True
    assert _queue(runtime).read() == []


async def test_promoting_something_that_is_gone_says_so(memory_tool: Memory) -> None:
    with tool_call_context("Memory"):
        result = await _op(memory_tool, op="promote", id="deadbeef")

    assert result.is_error
    assert "deadbeef" in str(result)
