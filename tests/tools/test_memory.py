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
