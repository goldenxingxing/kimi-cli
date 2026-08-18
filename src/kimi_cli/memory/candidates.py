"""Facts noticed automatically, waiting to be approved.

Persistent memory only ever contained what the agent thought to record with a
tool call, which means anything it did not think of was simply lost — the one
structural gap against systems that extract from the conversation themselves.

Those systems close it by writing what they extract. This does not: a
candidate is a *proposal*, held in its own file, and nothing reaches persistent
memory without the same approval an explicit ``add`` requires. What is
automated is the noticing, not the deciding.

Candidates are cheap and disposable. They expire, they are capped, and losing
the file costs nothing that was not already in the conversation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator

from kimi_cli.memory.entry import MemoryEntry, MemoryKind
from kimi_cli.utils.logging import logger

CANDIDATES_FILENAME = "candidates.jsonl"

#: Kept small on purpose. A backlog nobody clears is noise in every future
#: session, and the newest proposals are the ones still worth deciding on.
MAX_CANDIDATES = 12

#: A proposal nobody acted on for a fortnight was not worth acting on.
CANDIDATE_TTL_SECONDS = 14 * 24 * 3600


class MemoryCandidate(BaseModel):
    """A fact the archivist thinks is worth keeping, pending approval."""

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    kind: MemoryKind
    content: str
    key: str | None = None
    created_at: float = Field(default_factory=time.time)
    session_id: str | None = None

    @field_validator("key")
    @classmethod
    def _check_key(cls, value: str | None) -> str | None:
        """Hold a proposed key to the rule the stored entry will apply.

        A key that only fails on promotion turns an approval the user has
        already given into an error, so an unusable one is dropped here and the
        fact is kept.
        """
        if not value:
            return None
        try:
            return MemoryEntry(kind="project", scope="session", content="x", key=value).key
        except ValueError:
            return None

    def render_index(self, *, width: int = 96) -> str:
        first = next((ln.strip() for ln in self.content.splitlines() if ln.strip()), "")
        if len(first) > width:
            first = first[: width - 1].rstrip() + "…"
        return f"- [{self.kind}] ({self.id}) {first}"


@dataclass(frozen=True, slots=True)
class CandidateFile:
    """Read/write access to one user's pending proposals."""

    path: Path

    def read(self) -> list[MemoryCandidate]:
        """Live candidates, oldest first. Unreadable lines are skipped."""
        if not self.path.exists():
            return []
        cutoff = time.time() - CANDIDATE_TTL_SECONDS
        out: list[MemoryCandidate] = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("could not read memory candidates", exc_info=True)
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                candidate = MemoryCandidate.model_validate_json(line)
            except ValidationError:
                continue
            if candidate.created_at >= cutoff:
                out.append(candidate)
        return out[-MAX_CANDIDATES:]

    def write(self, candidates: list[MemoryCandidate]) -> None:
        """Replace the file. Never raises — a lost proposal is not an outage."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            body = "\n".join(
                json.dumps(c.model_dump(mode="json"), ensure_ascii=False)
                for c in candidates[-MAX_CANDIDATES:]
            )
            self.path.write_text(body + "\n" if body else "", encoding="utf-8")
        except OSError:
            logger.warning("could not write memory candidates", exc_info=True)

    def add(self, new: list[MemoryCandidate]) -> None:
        """Append proposals, dropping ones that restate what is already queued."""
        existing = self.read()
        seen = {c.content.strip().casefold() for c in existing}
        fresh = [c for c in new if c.content.strip().casefold() not in seen]
        if fresh:
            self.write(existing + fresh)

    def take(self, candidate_id: str) -> MemoryCandidate | None:
        """Remove and return one proposal, by id."""
        wanted = candidate_id.strip().lower()
        remaining: list[MemoryCandidate] = []
        found: MemoryCandidate | None = None
        for candidate in self.read():
            if found is None and candidate.id.lower() == wanted:
                found = candidate
            else:
                remaining.append(candidate)
        if found is not None:
            self.write(remaining)
        return found
