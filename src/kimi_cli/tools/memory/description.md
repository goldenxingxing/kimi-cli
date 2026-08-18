Manage structured memory across the agent's lifetime.

Use this tool to record durable, non-obvious knowledge that should outlive a
single conversational turn — and, optionally, the current session.

Memory is split into two scopes:

- **`session`** — local to the current conversation. Cleared when the session
  ends. Use this for working notes and short-lived state.
- **`persistent`** — written to the user's private memory file
  (`<share>/users/<owner>/memory/persistent.jsonl`) and surfaced in *future*
  sessions of this same user via a system reminder. Use sparingly — only for
  knowledge that is genuinely worth remembering across sessions.

Each entry has an `id` (returned on `add`), a `kind`, a `scope`, and a
`content` body. Use `update` / `delete` with the `id` to revise or remove
entries.

Adding a persistent fact that restates one already stored does not create a
second entry: it updates the existing one in place and `add` returns
`result: "merged"` along with `merged_into` (the surviving id) and `replaced`
(the wording it overwrote). A fact that differs in a **number, an identifier,
or a negation** is never treated as a restatement, so correcting a date or
flipping a rule always produces a separate entry — delete the stale one
yourself if it should not survive.

## Kinds — what to save and when

Pick exactly one kind per entry:

- **`user`** — the user's role, goals, knowledge background, or preferences.
  Save when you learn who the user is or how they like to work.
  *Examples:* "User is a senior backend engineer focused on observability",
  "Prefers terse responses with no trailing summary", "New to React, deep Go
  background — frame frontend explanations via backend analogues".

- **`feedback`** — corrections or affirmations the user gave about agent
  behavior. Save both when the user says "don't do X" and when they confirm a
  non-obvious choice with "yes, exactly that". Lead with the rule, then a
  short *Why* (the reason given) and *How to apply* (when it kicks in) so
  future-you can judge edge cases.
  *Examples:* "Don't mock the database in integration tests — last quarter a
  mocked test passed but the prod migration broke", "Bundle related refactors
  into one PR — confirmed when I made that call here".

- **`project`** — current project state, goals, incidents, or decision
  context that **is not derivable from code or git history**: who is doing
  what, why, by when. Convert relative dates to absolute dates before
  saving (e.g. "Thursday" → the actual ISO date).
  *Examples:* "Merge freeze begins 2026-03-05 for mobile release cut",
  "Auth middleware rewrite is driven by legal/compliance, not tech debt —
  prefer compliance over ergonomics in scope decisions".

- **`reference`** — pointers to where information lives in external systems
  (doc URLs, dashboards, ticket trackers, file paths). The value is the
  pointer plus a one-line note about *what is there*, not a copy of the
  contents.
  *Examples:* "Pipeline bugs are tracked in Linear project INGEST",
  "grafana.internal/d/api-latency is the oncall latency dashboard — check it
  when editing request-path code".

## What NOT to save

These are explicitly out of scope. Do not save them even when asked:

- **Code patterns, architecture, file paths, project structure** — derivable
  by reading the current code.
- **Git history, recent changes, who-changed-what** — `git log` / `git blame`
  are authoritative.
- **Debugging solutions or fix recipes** — the fix is in the code; the commit
  message has the rationale.
- **Anything already documented in `AGENTS.md`** — those files are loaded
  into context already.
- **Ephemeral task or session state** — current work-in-progress, temporary
  variables, in-flight conversation context. Use the `session` scope for
  short-lived working notes if you must.

If the user asks you to save something that falls into the categories above,
ask them what was *surprising* or *non-obvious* about it — that is the part
worth keeping.

## When to read

Call `list` at the start of a turn if you need to recall what was previously
saved, or to confirm a fact before relying on it. The system also injects
persistent entries automatically — but not all of them in full.

## What the opening snapshot contains

`user` and `feedback` entries appear in full: they describe how to work, and
would not change anything sitting behind a lookup you have no reason to
perform.

`project` and `reference` entries appear as a one-line index, each with a
handle in parentheses. Read one in full with `get` when its summary looks
relevant to the task in front of you. Do not guess at the rest of an entry
from its summary line — fetch it.

## Handles

Give `project` and `reference` entries a `key` when you add them: a short
`namespace/slug` such as `acls/repo-path`, grouping related entries under one
namespace. It becomes the handle a later session sees in the index, and it is
what makes an entry recognisable there — an entry without one can only show an
opaque id fragment.

## Operations

- `add(kind, scope, content, key=None)` → returns the new entry's `id`
- `get(handle)` → the full entry, addressed by `key` or `id`
- `list(scope)` → returns formatted entries (use `scope="all"` for everything)
- `update(id, content)` → replace the body of an existing entry
- `delete(id)` → remove an entry by id
