"""The web memory API writes to the same file as the tool, under the same rules."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kimi_cli.memory.candidates import CANDIDATES_FILENAME, CandidateFile, MemoryCandidate
from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.storage import read_entries, upsert_entry
from kimi_cli.web.api import memory
from kimi_cli.web.user_auth import get_current_user


@pytest.fixture
def persistent_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "persistent.jsonl"
    monkeypatch.setattr(memory, "get_persistent_memory_file", lambda _owner: path)
    return path


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(memory.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "u1", "role": "user"}
    with TestClient(app) as test_client:
        yield test_client


def add(client: TestClient, content: str, kind: str = "user"):
    return client.post("/api/memory/persistent", json={"kind": kind, "content": content})


def test_repeating_a_fact_updates_it_and_stops_claiming_creation(
    client: TestClient, persistent_path: Path
) -> None:
    first = add(client, "User prefers terse replies")
    assert first.status_code == 201
    assert first.json()["merged"] is False

    second = add(client, "User prefers terse replies.")

    assert second.status_code == 200
    body = second.json()
    assert body["merged"] is True
    assert body["id"] == first.json()["id"]
    # The response must describe the record on disk, not the one we asked for.
    assert body["content"] == "User prefers terse replies."
    assert body["replaced"] == "User prefers terse replies"
    assert len(client.get("/api/memory/persistent").json()) == 1


def test_a_near_miss_is_created_and_reported(client: TestClient, persistent_path: Path) -> None:
    first = add(client, "Pipeline bugs tracked in Linear project INGEST")
    second = add(client, "Pipeline bugs tracked in Linear project EGRESS")

    assert second.status_code == 201
    duplicates = second.json()["possible_duplicates"]
    assert [d["id"] for d in duplicates] == [first.json()["id"]]
    assert len(client.get("/api/memory/persistent").json()) == 2


def test_a_merged_id_is_still_addressable(client: TestClient, persistent_path: Path) -> None:
    first = add(client, "User prefers terse replies").json()
    add(client, "User prefers terse replies.")

    updated = client.put(f"/api/memory/persistent/{first['id']}", json={"content": "Revised"})

    assert updated.status_code == 200
    assert updated.json()["content"] == "Revised"


def test_the_tool_and_the_api_share_one_invariant(
    client: TestClient, persistent_path: Path
) -> None:
    # Writing the same fact through each door must not produce two entries;
    # otherwise the store's shape depends on which one was used.
    add(client, "User prefers terse replies")
    upsert_entry(
        persistent_path,
        MemoryEntry(kind="user", scope="persistent", content="User prefers terse replies."),
    )

    assert len(read_entries(persistent_path)) == 1


@pytest.fixture
def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CandidateFile:
    """The suggestion queue this caller's endpoints act on."""
    directory = tmp_path / "memory"
    directory.mkdir()
    monkeypatch.setattr(memory, "get_user_memory_dir", lambda _owner: directory)
    return CandidateFile(directory / CANDIDATES_FILENAME)


def suggest(queue: CandidateFile, content: str, kind: str = "project") -> MemoryCandidate:
    candidate = MemoryCandidate(kind=kind, content=content)
    queue.add([candidate])
    return candidate


class TestSuggestionsCanBeDecidedWithoutAskingTheAgent:
    """The agent has had promote and dismiss since the tool existed, and the
    preamble lists what is waiting — but it raises a suggestion only when the
    subject comes up. Anything off-topic stayed queued until it expired, with
    no way for its owner to see it. Five real proposals sat for two days that
    way.
    """

    def test_the_queue_is_visible(self, client: TestClient, queue: CandidateFile) -> None:
        suggest(queue, "Reports live under output/reports/daily")

        listed = client.get("/api/memory/candidates").json()

        assert [c["content"] for c in listed] == ["Reports live under output/reports/daily"]

    def test_promoting_stores_it_and_clears_the_queue(
        self, client: TestClient, queue: CandidateFile, persistent_path: Path
    ) -> None:
        candidate = suggest(queue, "Ask before emailing a client", kind="feedback")

        response = client.post(f"/api/memory/candidates/{candidate.id}/promote")

        assert response.status_code == 201
        assert [e.content for e in read_entries(persistent_path)] == [
            "Ask before emailing a client"
        ]
        assert queue.read() == []

    def test_a_failed_write_leaves_the_suggestion_to_decide_again(
        self, client: TestClient, queue: CandidateFile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Written first, dequeued second. The other order drops it silently."""
        candidate = suggest(queue, "Ask before emailing a client")

        def _fail(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(memory, "upsert_entry", _fail)
        with pytest.raises(OSError, match="disk full"):
            client.post(f"/api/memory/candidates/{candidate.id}/promote")

        assert [c.id for c in queue.read()] == [candidate.id]

    def test_dismissing_drops_it_and_writes_nothing(
        self, client: TestClient, queue: CandidateFile, persistent_path: Path
    ) -> None:
        candidate = suggest(queue, "A guess nobody wants kept")

        assert client.delete(f"/api/memory/candidates/{candidate.id}").status_code == 204
        assert queue.read() == []
        assert read_entries(persistent_path) == []

    def test_an_id_that_is_gone_is_not_a_silent_success(
        self, client: TestClient, queue: CandidateFile
    ) -> None:
        assert client.post("/api/memory/candidates/deadbeef/promote").status_code == 404
        assert client.delete("/api/memory/candidates/deadbeef").status_code == 404
