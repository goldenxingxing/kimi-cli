# Session List Network Retry Design

## Goal

Prevent a transient browser-to-local-backend connection failure from producing a persistent
session error toast when a subsequent retry can load the session list successfully.

## Scope

The change applies only to full active-session list reads performed by `refreshSessions`, including
initial load, search refresh, periodic refresh, and refresh after the page becomes visible.

The following operations remain unchanged:

- pagination through `loadMoreSessions`;
- archived-session reads;
- session creation, deletion, rename, archive, unarchive, fork, and bulk mutations;
- all other API clients and endpoints.

## Design

Add a small framework-independent retry helper outside the generated OpenAPI client. The helper
accepts an asynchronous operation, retries it once after a one-second delay only when the thrown
value is an OpenAPI `FetchError`, and otherwise rethrows immediately.

`refreshSessions` will call the sessions list endpoint through this helper. A successful retry is
treated exactly like a successful first request: it updates the sessions and pagination state and
does not set the hook error.

If both attempts fail, the helper rethrows the final error. `refreshSessions` logs that error for
diagnostics. When the final error is a `FetchError`, the hook exposes a localized
"temporarily unable to connect to the local service" message instead of the generated client's
interceptor message. HTTP errors, validation errors, parsing errors, and other non-network errors
retain their existing messages and are never retried.

The generated `web/src/lib/api/runtime.ts` file will not be edited.

## Localization

Add a session-network-error description to every existing toast locale. English communicates that
the local service is temporarily unavailable and asks the user to retry shortly. Chinese conveys
the same meaning. Other supported locales receive an equivalent translation consistent with the
existing locale structure.

## Testing

Use Node's built-in test runner against the framework-independent helper, without adding a new test
framework dependency. Tests cover:

1. a `FetchError` is retried once and a successful second result is returned;
2. a non-`FetchError` is not retried;
3. a second `FetchError` is rethrown after exactly two attempts;
4. the injected delay runs only between the first and second attempts.

After the focused tests pass, run the Web project's type check, lint, and production build.

## Success Criteria

- A one-off `FetchError` during a full session-list refresh produces no toast when the retry works.
- A persistent network failure produces one localized session error after the retry fails.
- HTTP and business errors preserve their current behavior.
- No session write operation is retried.
- Generated OpenAPI source remains unchanged.
