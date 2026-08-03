"""Tests for bounded, fail-open Wiki retrieval at root-turn start."""

from __future__ import annotations

from uuid import uuid4

import pytest

from kimi_cli.wiki.search import SearchResult
from kimi_cli.wiki.triggers import WikiTurnCoordinator


@pytest.fixture
def coordinator() -> WikiTurnCoordinator:
    return WikiTurnCoordinator(provenance_session_id=uuid4())


class _Manager:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self.queries.append((query, limit))
        return self.results


def _result(index: int, *, snippet: str = "durable retrieval reference") -> SearchResult:
    return SearchResult(
        logical_path=f"concepts/reference-{index}.md",
        title=f"Reference {index}",
        summary="Stable summary",
        snippet=snippet,
        score=1.0,
        revision=index,
        content_hash=f"sha256:{index:064x}",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["atomic wiki recovery", "全局知识检索为什么没有触发"])
async def test_real_root_prompt_retrieves_at_most_three_bounded_pages(
    coordinator: WikiTurnCoordinator, query: str
) -> None:
    from kimi_cli.wiki.retrieval import (
        OPENKIMO_WIKI_RETRIEVAL_START,
        RETRIEVAL_MAX_BLOCK_BYTES,
        RETRIEVAL_MAX_SNIPPET_BYTES,
        retrieve_for_turn,
    )

    manager = _Manager([_result(index, snippet="检索结果" * 300) for index in range(6)])
    await coordinator.begin_turn(query, query.strip())

    result = await retrieve_for_turn(manager, coordinator, query)

    assert result is not None
    assert result.result_count == 3
    assert manager.queries == [(query, 3)]
    assert len(result.block.encode("utf-8")) <= RETRIEVAL_MAX_BLOCK_BYTES
    assert result.block.count(OPENKIMO_WIKI_RETRIEVAL_START) == 1
    assert all(
        len(line.removeprefix("snippet: ").encode("utf-8")) <= RETRIEVAL_MAX_SNIPPET_BYTES
        for line in result.block.splitlines()
        if line.startswith("snippet: ")
    )
    result.block.encode("utf-8").decode("utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "reason", "kwargs"),
    [
        ("/help", "slash_command", {"slash_command": True}),
        ("a", "too_short", {}),
        ("api_key=secret-value", "sensitive", {}),
        ("normal prompt", "synthetic", {"synthetic": True}),
    ],
)
async def test_retrieval_skips_non_user_or_sensitive_input(
    coordinator: WikiTurnCoordinator,
    text: str,
    reason: str,
    kwargs: dict[str, bool],
) -> None:
    from kimi_cli.wiki.retrieval import retrieve_for_turn

    manager = _Manager([_result(1)])

    assert await retrieve_for_turn(manager, coordinator, text, **kwargs) is None
    assert manager.queries == []
    assert coordinator.last_retrieval_outcome == reason


@pytest.mark.asyncio
async def test_empty_wiki_and_zero_hits_inject_nothing(coordinator: WikiTurnCoordinator) -> None:
    from kimi_cli.wiki.retrieval import retrieve_for_turn

    manager = _Manager([])
    await coordinator.begin_turn("durable architecture", "durable architecture")

    assert await retrieve_for_turn(manager, coordinator, "durable architecture") is None
    assert coordinator.last_retrieval_outcome == "empty"


def test_query_normalization_truncates_only_complete_utf8_code_points() -> None:
    from kimi_cli.wiki.retrieval import build_retrieval_query

    assert build_retrieval_query("  a\n\t b  ") == "a b"
    assert build_retrieval_query("检索" * 10, max_bytes=10) == "检索检"
