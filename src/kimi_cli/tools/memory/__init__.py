import json
from pathlib import Path
from typing import Literal, cast, override

from kosong.tooling import BriefDisplayBlock, CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field, field_validator

from kimi_cli.memory import (
    DuplicateVerdict,
    MemoryEntry,
    UpsertResult,
    classify_entry,
    delete_entry,
    read_entries,
    update_entry,
    upsert_entry,
)
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.utils import load_desc

NAME = "Memory"

_BASE_DESCRIPTION = load_desc(Path(__file__).parent / "description.md")

ListScope = Literal["session", "persistent", "all"]
WriteScope = Literal["session", "persistent"]
EntryKind = Literal["user", "feedback", "project", "reference"]


class AddOp(BaseModel):
    op: Literal["add"] = "add"
    kind: EntryKind = Field(description="The category of memory being recorded.")
    scope: WriteScope = Field(
        description=(
            "`session` keeps the entry in the current conversation only. "
            "`persistent` writes to the user's cross-session memory."
        ),
    )
    content: str = Field(min_length=1, description="The memory body. Be concise but specific.")


class ListOp(BaseModel):
    op: Literal["list"] = "list"
    scope: ListScope = Field(default="all", description="Which scope(s) to list.")


class UpdateOp(BaseModel):
    op: Literal["update"] = "update"
    id: str = Field(description="The id of the entry to update.")
    content: str = Field(min_length=1, description="The new body for the entry.")


class DeleteOp(BaseModel):
    op: Literal["delete"] = "delete"
    id: str = Field(description="The id of the entry to delete.")


class Params(BaseModel):
    operation: AddOp | ListOp | UpdateOp | DeleteOp = Field(
        discriminator="op",
        description="The memory operation to perform.",
    )

    @field_validator("operation", mode="before")
    @classmethod
    def parse_json_operation(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return cast(dict[str, object], decoded) if isinstance(decoded, dict) else value


def _ok(output: str, brief: str) -> ToolReturnValue:
    return ToolReturnValue(
        is_error=False,
        output=output,
        message="",
        display=[BriefDisplayBlock(text=brief)],
    )


def _preview(content: str) -> str:
    return content if len(content) <= 200 else content[:200] + "..."


def _add_payload(result: UpsertResult, op: AddOp) -> dict[str, object]:
    """The tool result.

    ``id``/``scope``/``kind`` are unchanged so anything reading those keeps
    working. ``result`` is what lets the model tell a create from a merge — the
    id alone cannot, since on a merge it belongs to an entry the model may
    never have seen. ``replaced`` is the recovery channel: if the merge was
    wrong, the model can restore the old wording in the same turn.
    """
    payload: dict[str, object] = {
        "id": result.entry.id,
        "scope": "persistent",
        "kind": op.kind,
        "result": "merged" if result.merged else "created",
    }
    if result.merged:
        payload["merged_into"] = result.entry.id
        payload["similarity"] = round(result.score, 3)
        if result.replaced_content is not None:
            payload["replaced"] = _preview(result.replaced_content)
    if result.advisories:
        payload["possible_duplicates"] = [
            {"id": entry_id, "similarity": round(score, 3), "content": _preview(content)}
            for entry_id, content, score in result.advisories
        ]
    return payload


def _describe_add(op: AddOp, verdict: DuplicateVerdict, existing: list[MemoryEntry]) -> str:
    """The approval prompt.

    A merge must show both texts: the older one is about to be destroyed, and
    this prompt is the only place a human can stop it.
    """
    if verdict.action != "merge" or verdict.target_id is None:
        return f"Add persistent memory ({op.kind}): {_preview(op.content)}"

    target = next((e for e in existing if e.id == verdict.target_id), None)
    head = (
        f"Merge persistent memory ({op.kind}) into {verdict.target_id[:8]}"
        f" — {verdict.score:.0%} match"
    )
    if target is None:
        return f"{head}\n  new:      {_preview(op.content)}"
    return f"{head}\n  existing: {_preview(target.content)}\n  new:      {_preview(op.content)}"


def _add_brief(result: UpsertResult, op: AddOp) -> str:
    if result.merged:
        return f"Merged into {result.entry.id[:8]} (persistent/{op.kind})"
    return f"Remembered (persistent/{op.kind})"


def _format_entries(entries: list[MemoryEntry], header: str) -> str:
    if not entries:
        return f"{header}: (empty)"
    lines = [f"{header}:"]
    for e in entries:
        lines.append(e.render())
    return "\n".join(lines)


class Memory(CallableTool2[Params]):
    name: str = NAME
    description: str = _BASE_DESCRIPTION
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._runtime = runtime

    @property
    def _persistent_file(self) -> Path:
        return self._runtime.user_memory_dir / "persistent.jsonl"

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        op = params.operation
        if isinstance(op, AddOp):
            return await self._add(op)
        if isinstance(op, ListOp):
            return self._list(op)
        if isinstance(op, UpdateOp):
            return await self._update(op)
        return await self._delete(op)

    async def _request_persistent_approval(
        self, action: str, description: str
    ) -> ToolReturnValue | None:
        """Gate persistent-memory mutations through user approval.

        Persistent memory survives across sessions and influences the agent's
        future behavior, so the user must opt in. Returns ``None`` when the
        action is approved (continue), or a rejection ``ToolError`` otherwise.

        ``action`` becomes part of the approval key that "always allow"
        remembers, so a merge must not share one with an add: allowing the
        agent to *create* entries is not consent to *overwrite* existing ones.
        """
        result = await self._runtime.approval.request(
            self.name,
            f"memory.{action}",
            description,
        )
        if not result:
            return result.rejection_error()
        return None

    async def _add(self, op: AddOp) -> ToolReturnValue:
        if op.scope == "session":
            entry = MemoryEntry(kind=op.kind, scope=op.scope, content=op.content)
            self._runtime.session.state.session_memory.append(entry)
            self._runtime.session.save_state()
            return _ok(
                output=json.dumps(
                    {"id": entry.id, "scope": "session", "kind": op.kind, "result": "created"}
                ),
                brief=f"Remembered (session/{op.kind})",
            )

        # Classify before asking, so the prompt can describe what a merge would
        # overwrite. The write below re-classifies under the lock and pins the
        # target, so a file that changes in between degrades to a create rather
        # than merging into an entry the user was never shown.
        existing = read_entries(self._persistent_file)
        verdict = classify_entry(op.content, op.kind, existing)
        rejection = await self._request_persistent_approval(
            "merge" if verdict.action == "merge" else "add",
            _describe_add(op, verdict, existing),
        )
        if rejection is not None:
            return rejection

        result = upsert_entry(
            self._persistent_file,
            MemoryEntry(kind=op.kind, scope="persistent", content=op.content),
            expect_target_id=verdict.target_id if verdict.action == "merge" else None,
        )
        return _ok(output=json.dumps(_add_payload(result, op)), brief=_add_brief(result, op))

    def _list(self, op: ListOp) -> ToolReturnValue:
        sections: list[str] = []
        if op.scope in ("session", "all"):
            sections.append(
                _format_entries(
                    list(self._runtime.session.state.session_memory),
                    "Session memory",
                )
            )
        if op.scope in ("persistent", "all"):
            sections.append(
                _format_entries(read_entries(self._persistent_file), "Persistent memory")
            )
        return _ok(output="\n\n".join(sections), brief=f"Listed ({op.scope})")

    async def _update(self, op: UpdateOp) -> ToolReturnValue:
        # Try session first (cheaper), then persistent.
        for i, entry in enumerate(self._runtime.session.state.session_memory):
            if entry.id == op.id:
                self._runtime.session.state.session_memory[i] = entry.model_copy(
                    update={"content": op.content}
                )
                self._runtime.session.save_state()
                return _ok(
                    output=json.dumps({"id": op.id, "scope": "session"}),
                    brief="Memory updated",
                )
        # Persistent path requires approval.
        rejection = await self._request_persistent_approval(
            "update",
            f"Update persistent memory ({op.id[:8]}): {_preview(op.content)}",
        )
        if rejection is not None:
            return rejection
        updated = update_entry(self._persistent_file, op.id, op.content)
        if updated is None:
            return ToolError(message=f"No memory entry with id={op.id!r}.", brief="Not found")
        return _ok(
            output=json.dumps({"id": op.id, "scope": "persistent"}),
            brief="Memory updated",
        )

    async def _delete(self, op: DeleteOp) -> ToolReturnValue:
        before = len(self._runtime.session.state.session_memory)
        self._runtime.session.state.session_memory[:] = [
            e for e in self._runtime.session.state.session_memory if e.id != op.id
        ]
        if len(self._runtime.session.state.session_memory) != before:
            self._runtime.session.save_state()
            return _ok(
                output=json.dumps({"id": op.id, "scope": "session"}),
                brief="Memory deleted",
            )
        # Persistent path requires approval.
        rejection = await self._request_persistent_approval(
            "delete",
            f"Delete persistent memory entry {op.id[:8]}",
        )
        if rejection is not None:
            return rejection
        if delete_entry(self._persistent_file, op.id):
            return _ok(
                output=json.dumps({"id": op.id, "scope": "persistent"}),
                brief="Memory deleted",
            )
        return ToolError(message=f"No memory entry with id={op.id!r}.", brief="Not found")
