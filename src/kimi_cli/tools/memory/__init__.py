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
from kimi_cli.memory.candidates import CANDIDATES_FILENAME, CandidateFile
from kimi_cli.memory.consolidate import BEHAVIOURAL_BUDGET_CHARS, LONG_ENTRY_CHARS
from kimi_cli.memory.entry import BEHAVIOURAL_KINDS
from kimi_cli.memory.search import MemorySearchIndex
from kimi_cli.memory.storage import AmbiguousHandleError, resolve_handle, set_retired
from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.utils import load_desc

NAME = "Memory"


def _validate_key(value: str | None) -> str | None:
    """Run a candidate key through the same rule the stored entry applies."""
    if value is None:
        return None
    return MemoryEntry(kind="project", scope="session", content="x", key=value).key


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
    key: str | None = Field(
        default=None,
        description=(
            "Short semantic handle, `namespace/slug` — e.g. `acls/repo-path`. "
            "Lets a later session recognise this entry from a one-line index and "
            "fetch it by name. Give one for `project` and `reference` entries, "
            "which are read back on demand; group related entries under the same "
            "namespace. Optional, but an entry without one can only be addressed "
            "by an opaque id."
        ),
    )

    @field_validator("key")
    @classmethod
    def _check_key(cls, value: str | None) -> str | None:
        # Validated here as well as on the entry so a bad key is rejected as a
        # tool-argument error the model can correct, rather than surfacing much
        # later as a write failure with nothing pointing at the cause.
        return _validate_key(value)


class PromoteOp(BaseModel):
    op: Literal["promote"] = "promote"
    id: str = Field(description="Id of the suggested memory to keep, from the suggestions list.")


class DismissOp(BaseModel):
    op: Literal["dismiss"] = "dismiss"
    id: str = Field(description="Id of the suggested memory to drop, from the suggestions list.")


class SearchOp(BaseModel):
    op: Literal["search"] = "search"
    query: str = Field(
        min_length=1,
        description=(
            "Free text to look for across stored memory. Matches substrings, so "
            "a distinctive fragment works better than a sentence. Returns "
            "handles and snippets; read a hit in full with `get`."
        ),
    )


class GetOp(BaseModel):
    op: Literal["get"] = "get"
    handle: str = Field(
        description=(
            "The `key` or id shown in parentheses in the memory index. Returns the entry in full."
        ),
    )


class ListOp(BaseModel):
    op: Literal["list"] = "list"
    scope: ListScope = Field(default="all", description="Which scope(s) to list.")


class UpdateOp(BaseModel):
    op: Literal["update"] = "update"
    id: str = Field(
        description=(
            "Handle of the entry to update — the `key` or id shown in parentheses "
            "in the memory index, or a full id. Prefer updating an entry in place "
            "over adding one that says it supersedes an older record."
        )
    )
    content: str = Field(min_length=1, description="The new body for the entry.")


class RetireOp(BaseModel):
    """Stop injecting an entry without losing it.

    Deleting is for something that was wrong. This is for something that was
    right and no longer applies — a convention that changed, a project that
    ended. The record stays and stays searchable; it just stops arriving in
    every conversation, which is the only cost a stale behavioural entry
    actually imposes.
    """

    op: Literal["retire"]
    handle: str


class RestoreOp(BaseModel):
    """Put a retired entry back into force."""

    op: Literal["restore"]
    handle: str


class DeleteOp(BaseModel):
    op: Literal["delete"] = "delete"
    id: str = Field(
        description=(
            "Handle of the entry to delete — the `key` or id shown in parentheses "
            "in the memory index, or a full id."
        )
    )


class Params(BaseModel):
    operation: (
        AddOp
        | GetOp
        | SearchOp
        | PromoteOp
        | DismissOp
        | ListOp
        | UpdateOp
        | RetireOp
        | RestoreOp
        | DeleteOp
    ) = Field(
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
    # Behavioural entries are carried into every later conversation, so their
    # length is a standing cost rather than a one-off one — and it is invisible
    # from the write side, which is where the decision is made. Measured on a
    # real store: eleven entries averaging 576 characters had taken 79% of the
    # space there is, and three of them restated one procedure that already
    # existed as a file.
    if op.kind in BEHAVIOURAL_KINDS and len(op.content) >= LONG_ENTRY_CHARS:
        share = len(op.content) / BEHAVIOURAL_BUDGET_CHARS * 100
        payload["budget_note"] = (
            f"{len(op.content)} chars, about {share:.0f}% of the space every future "
            "conversation reserves for behavioural memory. If this is a procedure, "
            "keep the procedure in a file and record the pointer and the trigger."
        )

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

    @property
    def _search_db(self) -> Path:
        """Derived index, beside the store it is derived from.

        Safe to delete: it rebuilds from the JSONL on the next search.
        """
        return self._runtime.user_memory_dir / "search.sqlite3"

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        op = params.operation
        if isinstance(op, AddOp):
            return await self._add(op)
        if isinstance(op, GetOp):
            return self._get(op)
        if isinstance(op, SearchOp):
            return self._search(op)
        if isinstance(op, PromoteOp):
            return await self._promote(op)
        if isinstance(op, DismissOp):
            return self._dismiss(op)
        if isinstance(op, ListOp):
            return self._list(op)
        if isinstance(op, RetireOp | RestoreOp):
            return await self._set_retired(op)
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
            # Not turn-ending: remembering something is a side errand the agent
            # started on its own while working. "Do not save that" is not "stop
            # what you are doing", and treating it as such left the actual task
            # half-done with no way to resume it.
            return result.rejection_error(stops_turn=False)
        return None

    async def _add(self, op: AddOp) -> ToolReturnValue:
        if op.scope == "session":
            entry = MemoryEntry(kind=op.kind, scope=op.scope, content=op.content, key=op.key)
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
            MemoryEntry(kind=op.kind, scope="persistent", content=op.content, key=op.key),
            expect_target_id=verdict.target_id if verdict.action == "merge" else None,
        )
        return _ok(output=json.dumps(_add_payload(result, op)), brief=_add_brief(result, op))

    def _get(self, op: GetOp) -> ToolReturnValue:
        """Return one entry in full, addressed by key or id.

        Both are accepted because the index shows whichever the entry has: a
        record written before keys existed can only offer its id.
        """
        candidates = list(self._runtime.session.state.session_memory) + read_entries(
            self._persistent_file
        )
        try:
            entry = resolve_handle(candidates, op.handle)
        except AmbiguousHandleError as exc:
            return ToolError(message=str(exc), brief="Ambiguous handle")
        if entry is not None:
            return _ok(
                output=json.dumps(
                    {
                        "id": entry.id,
                        "key": entry.key,
                        "kind": entry.kind,
                        "scope": entry.scope,
                        "content": entry.content,
                    },
                    ensure_ascii=False,
                ),
                brief=f"Read {entry.handle}",
            )
        return ToolError(
            message=(
                f"No memory entry with handle {op.handle!r}. "
                "Use the handle shown in parentheses in the memory index, "
                "or `list` to see what is stored."
            ),
            brief="Not found",
        )

    @property
    def _candidate_file(self) -> CandidateFile:
        return CandidateFile(self._runtime.user_memory_dir / CANDIDATES_FILENAME)

    async def _promote(self, op: PromoteOp) -> ToolReturnValue:
        """Keep a suggested memory — through the same approval as any add.

        The suggestion was produced automatically; the decision is not. Taken
        off the queue only once the write succeeds, so a refusal leaves it
        there to be raised again rather than silently discarding it.
        """
        pending = self._candidate_file.read()
        wanted = op.id.strip().lower()
        candidate = next((c for c in pending if c.id.lower() == wanted), None)
        if candidate is None:
            return ToolError(
                message=(
                    f"No suggested memory with id {op.id!r}. It may have been "
                    "promoted, dismissed, or expired."
                ),
                brief="Not found",
            )

        rejection = await self._request_persistent_approval(
            "add",
            f"Keep this suggested memory?\n\n[{candidate.kind}] {candidate.content}",
        )
        if rejection is not None:
            return rejection

        result = upsert_entry(
            self._persistent_file,
            MemoryEntry(
                kind=candidate.kind,
                scope="persistent",
                content=candidate.content,
                key=candidate.key,
            ),
        )
        self._candidate_file.take(candidate.id)
        return _ok(
            output=json.dumps(
                {"id": result.entry.id, "key": result.entry.key, "promoted": True},
                ensure_ascii=False,
            ),
            brief=f"Kept {result.entry.handle}",
        )

    def _dismiss(self, op: DismissOp) -> ToolReturnValue:
        """Drop a suggestion. No approval — nothing was stored to begin with."""
        taken = self._candidate_file.take(op.id)
        if taken is None:
            return ToolError(message=f"No suggested memory with id {op.id!r}.", brief="Not found")
        return _ok(
            output=json.dumps({"id": taken.id, "dismissed": True}, ensure_ascii=False),
            brief="Dismissed",
        )

    def _search(self, op: SearchOp) -> ToolReturnValue:
        """Find entries by content when the index summary is not enough.

        The opening snapshot lists facts one line each, which is enough to
        recognise something already anticipated and not enough to find
        something half-remembered. This covers the second case.
        """
        entries = list(self._runtime.session.state.session_memory) + read_entries(
            self._persistent_file
        )
        index = MemorySearchIndex(self._search_db, self._persistent_file)
        hits = index.search(op.query, entries)
        if not hits:
            return _ok(
                output=json.dumps({"query": op.query, "hits": []}, ensure_ascii=False),
                brief="No matches",
            )
        return _ok(
            output=json.dumps(
                {
                    "query": op.query,
                    "hits": [
                        {"handle": h.handle, "kind": h.kind, "snippet": h.snippet} for h in hits
                    ],
                },
                ensure_ascii=False,
            ),
            brief=f"{len(hits)} match(es)",
        )

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
        try:
            updated = update_entry(self._persistent_file, op.id, op.content)
        except AmbiguousHandleError as exc:
            return ToolError(message=str(exc), brief="Ambiguous handle")
        if updated is None:
            return ToolError(message=f"No memory entry with id={op.id!r}.", brief="Not found")
        return _ok(
            output=json.dumps({"id": op.id, "scope": "persistent"}),
            brief="Memory updated",
        )

    async def _set_retired(self, op: RetireOp | RestoreOp) -> ToolReturnValue:
        """Take an entry out of the injected set, or put it back.

        Goes through the same approval as any other persistent write. Retiring
        does not destroy anything, but it does change what every later session
        is told without being asked — which is the part the user has a stake in.
        """
        retiring = isinstance(op, RetireOp)
        verb = "Retire" if retiring else "Restore"
        rejection = await self._request_persistent_approval(
            "retire" if retiring else "restore",
            f"{verb} persistent memory entry {op.handle}",
        )
        if rejection is not None:
            return rejection
        try:
            entry = set_retired(self._persistent_file, op.handle, retired=retiring)
        except AmbiguousHandleError as exc:
            return ToolError(message=str(exc), brief="Ambiguous handle")
        if entry is None:
            return ToolError(
                message=f"No memory entry with handle={op.handle!r}.", brief="Not found"
            )
        return _ok(
            output=json.dumps({"handle": entry.handle, "retired": entry.retired_at is not None}),
            brief=f"Memory {'retired' if retiring else 'restored'}",
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
        try:
            deleted = delete_entry(self._persistent_file, op.id)
        except AmbiguousHandleError as exc:
            return ToolError(message=str(exc), brief="Ambiguous handle")
        if deleted:
            return _ok(
                output=json.dumps({"id": op.id, "scope": "persistent"}),
                brief="Memory deleted",
            )
        return ToolError(message=f"No memory entry with id={op.id!r}.", brief="Not found")
