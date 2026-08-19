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

## Looking something up

Most of this page is about writing, and reading is the half that gets
forgotten: measured across real sessions, `search` was called zero times while
something in the store would have helped on roughly two thirds of turns.

Search before, not after:

- **Before re-deriving.** About to read files, grep, or run a command to
  establish how something here works, where something lives, or what was
  decided? Search first. One call against a small store is cheaper than the
  tool calls it saves, and the answer is the one this user already approved.
- **When the user refers to something as settled** — "the path we agreed",
  "like last time", "the convention for this" — search for it rather than
  asking them to repeat it.
- **Before saying you do not know** anything about this user's projects,
  conventions or history.

The snapshot is not the store. Behavioural entries arrive in full, but recorded
facts arrive as an index, and past a few hundred entries that index shows a
fraction of what exists and says so in its heading. **Not being listed is not
evidence it was never recorded.**

What not to do: searching on every turn. Most turns are instructions with
nothing to look up, and a store this size returns something for almost any
query — of what comes back for an arbitrary turn, roughly one in six helps.
Search when one of the triggers above fires, not as a reflex.

## How long an entry should be

A `user` or `feedback` entry is carried into every later conversation whether
or not it is relevant, so its length is a standing cost. About eight thousand
characters exist for all of them combined — at one sentence each that holds
fifty-odd rules, at a paragraph each it holds fourteen, and past the limit the
oldest stop arriving with nothing to show it.

**A procedure belongs in a file. Memory holds the pointer and the trigger.**

Measured on a real store: three entries totalling 2,900 characters — 36% of the
whole budget — were three versions of one daily-report procedure, and each of
them named the file where that procedure already lived. What was needed was one
line: where the file is, and when to open it.

    ✗  "Daily report SOP. (1) FIRST read /path/SOP.md. (2) Then scan
        session-data/ for … (3) For each session, read the first user
        message … (4) …"  — 1,038 characters, restating a file

    ✓  "Daily reports follow /path/SOP.md — read it before writing one;
        the three-scan rule in §4.0 is mandatory."  — 90 characters

Write the entry so a later session knows *that this constraint exists* and
where to look. It does not need to be able to reconstruct the procedure from
memory alone, because the file is right there.

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
handle and a date in parentheses. Read one in full with `get` when its summary
looks relevant to the task in front of you. Do not guess at the rest of an
entry from its summary line — fetch it. The date is when the entry was last
true: prefer the more recent of two entries that cover the same ground.

If you half-remember something that is not in the index — or the index has
been truncated — use `search` before concluding it was never recorded.

When a fact you already have recorded has changed, `update` it in place
instead of adding a new entry that says it supersedes the old one. The handle
shown in the index is what `update`, `retire` and `delete` accept.

## Retiring

`delete` is for an entry that was wrong. `retire` is for one that was right and
no longer applies — a convention that changed, a project that ended. A retired
entry stops being carried into new conversations but stays in the file and
stays reachable through `search`, and `restore` puts it back.

This matters because behavioural entries are injected whether or not anyone
asks for them, and only about a hundred fit. Past that the oldest stop arriving
on their own, so a store nobody prunes ends up with standing instructions that
are on disk and not in force. Retiring is how that happens on purpose. Both
operations ask for the user's approval, and neither removes anything.

## Suggested memories

The snapshot may also list suggestions — facts noticed automatically at the end
of an earlier conversation, which nobody has decided on yet. They are **not**
stored and **not** established fact; treat them as questions, not answers.

When one is relevant to what you are doing, say so and let the user decide,
then `promote` it to keep it or `dismiss` it to drop it. Promoting asks for the
same approval an explicit `add` does. Do not promote in bulk to tidy the list —
an unwanted memory costs more than a missing one, every session, forever.

## Handles

Give `project` and `reference` entries a `key` when you add them: a short
`namespace/slug` such as `acls/repo-path`, grouping related entries under one
namespace. It becomes the handle a later session sees in the index, and it is
what makes an entry recognisable there — an entry without one can only show an
opaque id fragment.

## Operations

- `add(kind, scope, content, key=None)` → returns the new entry's `id`
- `get(handle)` → the full entry, addressed by `key` or `id`
- `search(query)` → handles and snippets for entries containing the text
- `promote(id)` → keep a suggested memory (asks for approval)
- `dismiss(id)` → drop a suggested memory
- `list(scope)` → returns formatted entries (use `scope="all"` for everything)
- `update(id, content)` → replace the body of an existing entry
- `retire(handle)` → stop injecting an entry, without losing it
- `restore(handle)` → put a retired entry back into force
- `delete(id)` → remove an entry by id
