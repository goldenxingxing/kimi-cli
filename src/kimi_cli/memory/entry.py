from __future__ import annotations

import re
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

MemoryKind = Literal["user", "feedback", "project", "reference"]
MemoryScope = Literal["session", "persistent"]

#: Kinds the agent must be told outright, versus ones it can look up.
#:
#: A preference about how to work only changes behaviour if it is in front of
#: the model — nothing prompts it to go and fetch "be thorough". A fact about a
#: project is the opposite: worth having available, wasteful to carry into
#: every unrelated conversation.
BEHAVIOURAL_KINDS: frozenset[str] = frozenset({"user", "feedback"})
LOOKUP_KINDS: frozenset[str] = frozenset({"project", "reference"})

#: ``namespace/slug`` — lowercase, digits, dot, dash, one optional slash.
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)?$")
_KEY_MAX_LEN = 64


class MemoryEntry(BaseModel):
    """A single structured memory record.

    Used both inside ``SessionState.session_memory`` (session-scoped notes) and
    inside the user-private ``persistent.jsonl`` file (cross-session memory).
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: MemoryKind
    scope: MemoryScope
    content: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float | None = None
    last_relevant_at: float | None = None
    """When this entry's subject last came up in a conversation.

    Not when it was last obeyed — that has no trace. A rule is followed by
    *not* doing something, so "never force-push to main" and "no function names
    in the docs" leave exactly as much evidence when honoured as when
    forgotten. Anything scored on compliance would retire prohibitions first,
    which are the entries least safe to lose.

    Topicality is observable where compliance is not: whether the subject came
    up at all. A rule about documents is dormant if nobody has touched a
    document in months, and dormant is not wrong — it is only the difference
    between a rule that still has work to do and one that does not. Which is
    why this decides nothing on its own and only ranks what to ask about.
    """

    retired_at: float | None = None
    """When this stopped being injected, if it has.

    Behavioural memory is carried into every conversation whether or not anyone
    asks for it, and the budget that holds it fits roughly a hundred entries.
    Past that, the oldest simply stop arriving — so a store that only grows
    ends up with standing instructions that are still on disk and no longer in
    force, with nothing to say which.

    Retiring is how an entry leaves that set deliberately instead of by
    attrition. Nothing is removed: the record stays in the file, `search` and
    `get` still reach it, and clearing this field puts it back. A memory the
    user can open is worth more than one that is tidy, so retirement marks
    rather than deletes.
    """

    key: str | None = None
    """Short semantic handle, e.g. ``acls/repo-path``.

    ``id`` stays the primary key — it is written into every stored record and
    is what update and delete resolve against, so it cannot be given meaning
    after the fact without invalidating what is already on disk. This is the
    handle a *model* uses instead: it can be read, grouped by prefix, and
    mistyped visibly, none of which is true of a random hex string.

    Optional, and older records have none. Anything that displays a handle
    falls back to the id.
    """

    @field_validator("key")
    @classmethod
    def _check_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        if not value:
            return None
        if len(value) > _KEY_MAX_LEN or not _KEY_RE.fullmatch(value):
            raise ValueError(
                "key must be lowercase 'namespace/slug' (letters, digits, . _ -), "
                f"at most {_KEY_MAX_LEN} characters"
            )
        return value

    @property
    def handle(self) -> str:
        """What to show, and what ``get`` accepts back."""
        return self.key or self.id[:8]

    @property
    def is_behavioural(self) -> bool:
        """Whether this must be stated outright rather than looked up."""
        return self.kind in BEHAVIOURAL_KINDS

    def render(self) -> str:
        return f"- [{self.kind}] ({self.handle}) {self.content}"

    def render_index(self, *, width: int = 96) -> str:
        """One line: enough to judge relevance, not enough to act on.

        The summary is the first line of the body — memory is written as a
        statement, so its opening clause is what the entry is about.

        The date is when this was last true, not when it was filed: project
        facts go stale, and two entries about the same thing are told apart by
        which is more recent. Without it the agent has been reading undated
        claims and, in a real store, working around that by writing "supersedes
        the older record" into the text.
        """
        first = next((ln.strip() for ln in self.content.splitlines() if ln.strip()), "")
        if len(first) > width:
            first = first[: width - 1].rstrip() + "…"
        stamp = time.strftime("%Y-%m-%d", time.localtime(self.updated_at or self.created_at))
        return f"- [{self.kind}] ({self.handle}, {stamp}) {first}"
