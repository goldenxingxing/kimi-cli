"""The web memory API writes to the same file as the tool, under the same rules."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
