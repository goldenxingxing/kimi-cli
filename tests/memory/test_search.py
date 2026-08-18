"""Retrieval over stored memory.

The opening snapshot lists facts one line each — enough to recognise something
already anticipated, not enough to find something half-remembered.
"""

from __future__ import annotations

from pathlib import Path

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.search import MemorySearchIndex


def _entries(*bodies: str) -> list[MemoryEntry]:
    return [
        MemoryEntry(kind="project", scope="persistent", content=b, key=f"p/{i}")
        for i, b in enumerate(bodies)
    ]


def _index(tmp_path: Path) -> tuple[MemorySearchIndex, Path]:
    source = tmp_path / "persistent.jsonl"
    source.write_text("seed", encoding="utf-8")
    return MemorySearchIndex(tmp_path / "search.sqlite3", source), source


def test_it_finds_a_phrase_inside_chinese_without_spaces(tmp_path: Path) -> None:
    """The reason the index is trigram-tokenised.

    Chinese is written without spaces, so the default tokeniser makes a whole
    sentence one token and every phrase inside it is unfindable.
    """
    index, _ = _index(tmp_path)
    hits = index.search("仓库路径", _entries("acls 项目：真实仓库路径为 /Users/x/acls"))
    assert [h.handle for h in hits] == ["p/0"]


def test_a_two_character_query_falls_back_rather_than_missing(tmp_path: Path) -> None:
    """Two characters is a whole word in Chinese and below what trigram indexes."""
    index, _ = _index(tmp_path)
    hits = index.search("邮箱", _entries("126 邮箱已接入"))
    assert [h.handle for h in hits] == ["p/0"]


def test_punctuation_in_a_query_is_not_read_as_syntax(tmp_path: Path) -> None:
    index, _ = _index(tmp_path)
    entries = _entries("the file core.py lives under src/")
    for query in ("core.py", 'a "quoted" thing', "src/"):
        index.search(query, entries)  # must not raise


def test_an_empty_or_whitespace_query_returns_nothing(tmp_path: Path) -> None:
    index, _ = _index(tmp_path)
    assert index.search("", _entries("something")) == []
    assert index.search("   ", _entries("something")) == []


def test_the_index_refreshes_when_the_store_changes(tmp_path: Path) -> None:
    """It is a cache over a file the user can edit by hand."""
    index, source = _index(tmp_path)
    assert index.search("alpha", _entries("alpha fact")) != []

    source.write_text("changed", encoding="utf-8")
    hits = index.search("bravo", _entries("bravo fact"))
    assert [h.handle for h in hits] == ["p/0"]


def test_a_corrupt_index_degrades_to_no_results(tmp_path: Path) -> None:
    """Search is an aid; an unusable index must not break the tool call."""
    db = tmp_path / "search.sqlite3"
    db.write_bytes(b"this is not a database")
    source = tmp_path / "persistent.jsonl"
    source.write_text("seed", encoding="utf-8")

    assert MemorySearchIndex(db, source).search("anything", _entries("x y z")) == []


def test_results_are_capped(tmp_path: Path) -> None:
    index, _ = _index(tmp_path)
    hits = index.search("common", _entries(*[f"common token {i}" for i in range(40)]), limit=5)
    assert len(hits) == 5


class TestQueryConstruction:
    """How the query is built turned out to matter more than anything indexed.

    Measured on LoCoMo — 1,982 questions with the answering turn labelled —
    matching the whole query as one phrase scored 0.0% recall at every depth,
    because a question never appears verbatim inside an answer. Splitting it
    into OR-ed terms scored 65.3% at recall@10, past a BM25 baseline's 57.9%.
    """

    def test_a_question_matches_a_turn_that_answers_it(self, tmp_path: Path) -> None:
        index, _ = _index(tmp_path)
        entries = _entries(
            "Caroline: I went to a LGBTQ support group yesterday and it was powerful.",
            "Melanie: nice weather today",
        )

        hits = index.search("When did Caroline go to the LGBTQ support group?", entries)

        assert [h.handle for h in hits][:1] == ["p/0"]

    def test_grammar_alone_matches_nothing(self, tmp_path: Path) -> None:
        """Otherwise "when did the" would return the entire store."""
        index, _ = _index(tmp_path)
        assert index.search("when did the", _entries("some content here")) == []

    def test_a_chinese_question_matches_its_answer(self, tmp_path: Path) -> None:
        """Chinese has no spaces and no segmenter here.

        The words that carry meaning are two characters — 邮箱, 配置 — which is
        below what a trigram index holds, so this only works via the scan.
        """
        index, _ = _index(tmp_path)
        entries = _entries("126 邮箱已接入并验证可用，配置在 ~/mail.env", "无关的一条记录")

        hits = index.search("邮箱是怎么配置的", entries)

        assert [h.handle for h in hits][:1] == ["p/0"]

    def test_a_path_like_term_is_not_split_or_read_as_syntax(self, tmp_path: Path) -> None:
        index, _ = _index(tmp_path)
        hits = index.search("where is core.py", _entries("the file core.py lives in src/"))
        assert [h.handle for h in hits] == ["p/0"]

    def test_an_entry_matching_more_of_the_query_ranks_first(self, tmp_path: Path) -> None:
        index, _ = _index(tmp_path)
        entries = _entries("只提到配置", "同时提到邮箱和配置两件事")

        hits = index.search("邮箱配置", entries)

        assert hits[0].handle == "p/1"
