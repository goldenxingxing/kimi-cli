# Approval Recovery and Memory Validation Design

## Problem

Persistent Memory mutations wait for user approval. If the initial `ApprovalRequest`
is not registered by the web client, the backend waits indefinitely while the web
stream watchdog repeatedly reconnects. The existing reconnect path re-emits pending
approvals as out-of-band JSON-RPC requests, but it does not give the client an
authoritative snapshot to reconcile against.

More specifically, reconnect happens while the agent turn is still streaming, and
`WireServer._handle_initialize()` currently rejects initialization in that state with
`INVALID_STATE`. The pending-approval reissue code is below that guard and therefore
never runs. The web client retries initialization a limited number of times while the
backend continues waiting for the approval, producing the observed reconnect loop.

Separately, models sometimes encode `Memory.operation` as a JSON string, or pass only
the operation name. Pydantic rejects these calls before the tool runs. The agent can
then incorrectly tell the user that the mutation succeeded despite receiving a tool
error.

## Goals

- Restore every unresolved approval after a WebSocket reconnect, including when the
  original approval event was lost.
- Preserve a usable approval while reconnecting, then reconcile it with server state.
- Accept recoverable string-encoded Memory operation objects without weakening
  validation of incomplete operations.
- Require the agent to treat tool errors as failures rather than claiming success.
- Cover both regressions with automated tests.

## Approval Recovery Protocol

The wire server will include an `approval_requests` array in the successful
`initialize` result. Each item uses the existing `ApprovalRequest` payload shape.
This makes initialization an authoritative pending-approval snapshot while retaining
the existing request push for compatibility with other clients.

Initialization will also be permitted while a turn is streaming. It is a connection
handshake and pending-state read, not a new agent operation. The client used here does
not register external tools or hooks during reconnect; if a future client attempts
runtime-mutating initialization fields while streaming, the server will reject those
fields rather than reject the read-only handshake.

The web client will process that snapshot before considering initialization complete:

1. Convert each item to the existing `ApprovalRequestEvent` representation.
2. Feed it through the normal approval event handler, preserving one rendering path.
3. Remove locally retained, unsubmitted approvals that are absent from the snapshot.
4. Keep submitted approvals untouched until their normal resolution event arrives.

Closing a WebSocket will no longer erase unsubmitted approvals. They remain visible
but cannot be submitted while disconnected; reconnect initialization either confirms
them or removes stale entries. Explicit session changes and teardown continue to
clear all session-scoped interaction state.

The snapshot is the correctness mechanism. A backend timeout is not added because
automatic rejection can destroy a legitimate request while the user is temporarily
away.

## Memory Argument Normalization

`Params` will use a Pydantic pre-validation hook for `operation`:

- A JSON string whose decoded value is an object is converted to that object.
- A bare operation name such as `"update"` remains invalid because required fields
  cannot be recovered safely.
- Invalid JSON, arrays, scalar JSON, and incomplete objects continue to produce
  validation errors.

This accepts the recoverable 12:25 call shape while preventing guessed IDs or content.
The generated schema remains the discriminated object union, so correct model calls
remain the preferred path.

## Agent Success Reporting

The default system prompt will explicitly state that an error tool result means the
operation did not happen. The agent must retry with corrected arguments when safe or
report the failure; it must not claim success without a successful tool result.

This is a behavioral guard, not a replacement for tool validation.

## Testing

- Backend wire test: initialization returns all pending approvals using the existing
  payload fields.
- Frontend unit test: snapshot reconciliation restores a lost approval and removes a
  stale unsubmitted approval.
- Memory unit tests: a JSON-string operation object is accepted and dispatched;
  incomplete and non-object strings are rejected.
- Prompt snapshot test: the error-result success-reporting rule remains present.
- Run focused tests first, then Python checks and web typecheck/lint.

## Non-goals

- Changing approval policy or automatically approving/rejecting requests.
- Adding sounds or new notification UI.
- Repairing historical tool results or rewriting existing session logs.
