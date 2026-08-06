"""The cross-session snapshot has to come back after compaction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.storage import upsert_entry
from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.dynamic_injections import cross_session_memory
from kimi_cli.soul.dynamic_injections.cross_session_memory import (
    CrossSessionMemoryInjectionProvider,
)


@pytest.fixture
def soul(runtime: Runtime) -> Any:
    """The provider only reaches for ``soul.runtime``."""

    class FakeSoul:
        def __init__(self, rt: Runtime) -> None:
            self.runtime = rt

    return FakeSoul(runtime)


@pytest.fixture
def stored_fact(runtime: Runtime) -> Path:
    path = runtime.user_memory_dir / "persistent.jsonl"
    upsert_entry(
        path,
        MemoryEntry(kind="user", scope="persistent", content="User prefers terse replies"),
    )
    return path


async def test_the_snapshot_returns_after_compaction(soul: Any, stored_fact: Path) -> None:
    # The snapshot is an ordinary history message, so compaction folds it into
    # the compaction summary. Without re-injecting, the agent spends the rest
    # of a long session with no cross-session memory at all.
    provider = CrossSessionMemoryInjectionProvider()

    first = await provider.get_injections([], cast(Any, soul))
    assert first and "User prefers terse replies" in first[0].content

    await provider.on_context_compacted()

    again = await provider.get_injections([], cast(Any, soul))
    assert again and "User prefers terse replies" in again[0].content


async def test_the_snapshot_is_not_re_read_on_every_step(
    soul: Any, stored_fact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The one-shot guard exists to keep the prompt cache intact mid-session.
    provider = CrossSessionMemoryInjectionProvider()
    await provider.get_injections([], cast(Any, soul))

    reads = 0
    original = cross_session_memory.read_entries

    def counting_read(path: Path):
        nonlocal reads
        reads += 1
        return original(path)

    monkeypatch.setattr(cross_session_memory, "read_entries", counting_read)

    for _ in range(3):
        await provider.get_injections([], cast(Any, soul))

    assert reads == 0
