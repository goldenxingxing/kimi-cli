# Approval Recovery and Memory Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably restore pending approvals after reconnect and prevent recoverable Memory argument encoding errors and false success reports.

**Architecture:** Make the wire `initialize` result the authoritative snapshot of pending approvals while retaining existing request pushes for compatibility. Extract frontend reconciliation into a pure helper, normalize only JSON-object strings at the Memory schema boundary, and add an explicit system-prompt rule for failed tool results.

**Tech Stack:** Python 3.12+, Pydantic 2, pytest/pytest-asyncio, TypeScript 5.9, React 19, Node 26 test runner.

## Global Constraints

- Do not automatically approve or reject approval requests.
- Do not add an approval timeout.
- Preserve the existing JSON-RPC approval request push for compatibility.
- Reject incomplete or ambiguous Memory operation strings.
- Use failing tests before production changes.

---

### Task 1: Authoritative Pending-Approval Snapshot

**Files:**
- Modify: `src/kimi_cli/wire/server.py`
- Modify: `tests/core/test_wire_server_steer.py`

**Interfaces:**
- Consumes: `ApprovalRuntime.list_pending() -> list[ApprovalRequestRecord]`
- Produces: `initialize.result["approval_requests"]`, a JSON array of serialized `ApprovalRequest` payloads.

- [ ] **Step 1: Write failing reconnect-initialize and snapshot tests**

Set `server._cancel_event` to represent a running turn, create a pending runtime
request, call `_handle_initialize`, and assert that initialization succeeds and:

```python
assert response.result["approval_requests"] == [
    {
        "type": "ApprovalRequest",
        "payload": {
            "id": "req-init-1",
            "tool_call_id": "call-init-1",
            "sender": "Memory",
            "action": "memory.add",
            "description": "remember preference",
            "display": [],
            "source_kind": "foreground_turn",
            "source_id": "turn-init-1",
            "agent_id": None,
            "subagent_type": None,
        },
    }
]
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
uv run pytest tests/core/test_wire_server_steer.py::test_initialize_returns_pending_approval_snapshot -q
```

Expected: failure because `approval_requests` is absent.

- [ ] **Step 3: Permit read-only initialization while streaming and serialize pending approvals once**

Replace the unconditional streaming rejection with validation that only rejects
runtime-mutating reconnect fields:

```python
if self._is_streaming and (msg.params.external_tools or msg.params.hooks):
    return JSONRPCErrorResponse(
        id=msg.id,
        error=JSONRPCErrorObject(
            code=ErrorCodes.INVALID_STATE,
            message="Cannot register external tools or hooks while an agent turn is in progress",
        ),
    )
```

Add a focused converter in `WireServer`:

```python
def _pending_approval_messages(self) -> list[JsonType]:
    if self._approval_runtime is None:
        return []
    return [
        cast(
            JsonType,
            {
                "type": "ApprovalRequest",
                "payload": ApprovalRequest(
                    id=request.id,
                    tool_call_id=request.tool_call_id,
                    sender=request.sender,
                    action=request.action,
                    description=request.description,
                    display=request.display,
                    source_kind=request.source.kind,
                    source_id=request.source.id,
                    agent_id=request.source.agent_id,
                    subagent_type=request.source.subagent_type,
                ).model_dump(mode="json"),
            },
        )
        for request in self._approval_runtime.list_pending()
    ]
```

Set `result["approval_requests"]` from this method and continue reissuing each payload through `_request_approval`.

- [ ] **Step 4: Verify focused and neighboring wire tests**

Run:

```bash
uv run pytest tests/core/test_wire_server_steer.py tests/core/test_approval_runtime.py -q
```

Expected: all pass.

### Task 2: Frontend Approval Snapshot Reconciliation

**Files:**
- Create: `web/src/lib/approval-snapshot.ts`
- Create: `web/src/lib/approval-snapshot.test.ts`
- Modify: `web/src/hooks/useSessionStream.ts`
- Modify: `web/package.json`

**Interfaces:**
- Consumes: `ApprovalRequestEvent[]`, retained local request IDs.
- Produces: `reconcileApprovalRequestIds(localIds, serverIds) -> { confirmed: Set<string>; stale: Set<string> }`.

- [ ] **Step 1: Write the failing pure-helper tests**

Use Node's built-in test runner:

```ts
import assert from "node:assert/strict";
import test from "node:test";
import { reconcileApprovalRequestIds } from "./approval-snapshot.ts";

test("restores server approvals and removes stale local approvals", () => {
  const result = reconcileApprovalRequestIds(
    new Set(["stale", "confirmed"]),
    new Set(["confirmed", "lost"]),
  );
  assert.deepEqual([...result.confirmed].sort(), ["confirmed", "lost"]);
  assert.deepEqual([...result.stale], ["stale"]);
});
```

Add `"test:unit": "node --experimental-strip-types --test src/**/*.test.ts"` to `web/package.json`.

- [ ] **Step 2: Verify the helper test fails**

Run:

```bash
npm run test:unit
```

Expected: module/function not found.

- [ ] **Step 3: Implement the pure reconciliation helper**

```ts
export function reconcileApprovalRequestIds(
  localIds: ReadonlySet<string>,
  serverIds: ReadonlySet<string>,
): { confirmed: Set<string>; stale: Set<string> } {
  return {
    confirmed: new Set(serverIds),
    stale: new Set([...localIds].filter((id) => !serverIds.has(id))),
  };
}
```

- [ ] **Step 4: Apply the initialize snapshot**

Extend the initialize result type locally with:

```ts
approval_requests?: ApprovalRequestEvent[];
```

In the initialize response handler:

1. Process every snapshot event through `processEvent(event, false, event.payload.id)`.
2. Compute stale IDs from unsubmitted local approvals and the snapshot.
3. Resolve stale tool-message UI back to `input-available` and delete stale map entries.
4. Do not remove submitted requests.
5. Remove `pendingApprovalRequestsRef.current.clear()` from `ws.onclose`; retain explicit clears used for session switching/disposal.

- [ ] **Step 5: Verify frontend tests and static checks**

Run:

```bash
npm run test:unit
npm run typecheck
npm run lint
```

Expected: all pass with no TypeScript or Biome diagnostics.

### Task 3: Memory Operation String Normalization

**Files:**
- Modify: `src/kimi_cli/tools/memory/__init__.py`
- Create: `tests/tools/test_memory.py`

**Interfaces:**
- Consumes: `Params.model_validate({"operation": value})`
- Produces: parsed discriminated operation when `value` is a JSON string containing an object.

- [ ] **Step 1: Write failing schema tests**

```python
def test_memory_params_accept_json_encoded_operation_object() -> None:
    params = Params.model_validate(
        {"operation": '{"op":"update","id":"memory-1","content":"new value"}'}
    )
    assert params.operation == UpdateOp(id="memory-1", content="new value")


@pytest.mark.parametrize("operation", ['"update"', "[]", "{broken", "update"])
def test_memory_params_reject_non_object_operation_strings(operation: str) -> None:
    with pytest.raises(ValidationError):
        Params.model_validate({"operation": operation})
```

- [ ] **Step 2: Verify the schema tests fail for the expected reason**

Run:

```bash
uv run pytest tests/tools/test_memory.py -q
```

Expected: encoded-object acceptance test fails validation.

- [ ] **Step 3: Add a before-validator**

```python
@field_validator("operation", mode="before")
@classmethod
def parse_json_operation(cls, value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded if isinstance(decoded, dict) else value
```

Leave Pydantic responsible for discriminated-union and required-field validation.

- [ ] **Step 4: Verify Memory and schema regression tests**

Run:

```bash
uv run pytest tests/tools/test_memory.py tests/tools/test_tool_schemas.py -q
```

Expected: all pass.

### Task 4: Prevent False Success After Tool Errors

**Files:**
- Modify: `src/kimi_cli/agents/default/system.md`
- Modify: `tests/core/test_default_agent.py`

**Interfaces:**
- Produces: an invariant in every default agent prompt: no success claim after an error result.

- [ ] **Step 1: Write the failing prompt assertion**

```python
assert (
    "A tool result marked as an error means the requested operation did not succeed"
    in agent.system_prompt
)
assert "MUST NOT claim success" in agent.system_prompt
```

- [ ] **Step 2: Verify the prompt test fails**

Run:

```bash
uv run pytest tests/core/test_default_agent.py::test_default_agent -q
```

Expected: assertion or inline snapshot failure.

- [ ] **Step 3: Add the prompt rule**

Immediately after the existing tool-results paragraph, add:

```markdown
A tool result marked as an error means the requested operation did not succeed. You
MUST NOT claim success after such a result. Correct the tool arguments and retry when
it is safe and unambiguous; otherwise report the failure to the user.
```

Update the inline snapshot to match.

- [ ] **Step 4: Verify the prompt test**

Run:

```bash
uv run pytest tests/core/test_default_agent.py -q
```

Expected: all pass.

### Task 5: Full Verification

**Files:**
- Verify all files changed in Tasks 1–4.

**Interfaces:**
- Produces: evidence that focused regressions and project checks pass.

- [ ] **Step 1: Run focused Python tests**

```bash
uv run pytest tests/core/test_wire_server_steer.py tests/core/test_approval_runtime.py tests/tools/test_memory.py tests/tools/test_tool_schemas.py tests/core/test_default_agent.py -q
```

- [ ] **Step 2: Run Python lint and type checks on changed modules**

```bash
uv run ruff check src/kimi_cli/wire/server.py src/kimi_cli/tools/memory/__init__.py tests/core/test_wire_server_steer.py tests/tools/test_memory.py tests/core/test_default_agent.py
uv run pyright src/kimi_cli/wire/server.py src/kimi_cli/tools/memory/__init__.py
```

- [ ] **Step 3: Run web verification**

```bash
cd web
npm run test:unit
npm run typecheck
npm run lint
```

- [ ] **Step 4: Inspect the final diff**

```bash
git diff --check
git status --short
git diff --stat
```

Expected: only planned source, test, prompt, package-script, and documentation changes.
