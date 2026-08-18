"""Recall of the memory search against LoCoMo's gold evidence.

Retrieval only, and scored without a model call: the dataset names the turn
each answer lives in, so a hit is a fact rather than a judgement.

Three query styles, because how the query is formed turns out to matter more
than anything in the index:

  whole question   what a naive caller sends
  keywords         content words only, which is what the tool asks for
  best keyword     the single most distinctive word — an upper bound

Every dialogue turn is one memory entry. A question counts as answered if the
gold turn comes back at all.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.search import MemorySearchIndex

DATA = Path(__file__).with_name("locomo10.json")
"""Download: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"""
TOP_K = (1, 5, 10)

_STOP = {
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "and",
    "or",
    "s",
    "about",
    "that",
    "this",
    "it",
    "he",
    "she",
    "they",
    "his",
    "her",
    "their",
    "have",
    "has",
    "had",
    "been",
    "will",
    "would",
    "can",
    "could",
    "go",
    "get",
    "say",
    "said",
    "tell",
    "told",
    "know",
    "think",
    "make",
    "made",
}


def keywords(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", question)
    return [w for w in words if w.casefold() not in _STOP and len(w) > 2]


def entries_for(sample: dict) -> tuple[list[MemoryEntry], dict[str, str]]:
    convo = sample["conversation"]
    entries: list[MemoryEntry] = []
    by_id: dict[str, str] = {}
    for name in sorted(k for k in convo if k.startswith("session_") and "date" not in k):
        for turn in convo[name]:
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            entry = MemoryEntry(
                kind="project", scope="persistent", content=f"{turn['speaker']}: {text}"
            )
            entries.append(entry)
            by_id[entry.id] = turn["dia_id"]
    return entries, by_id


def main() -> None:
    with DATA.open(encoding="utf-8") as handle:
        samples = json.load(handle)
    tmp = Path(tempfile.mkdtemp())
    styles = ("whole question", "keywords", "best keyword")
    scores = {s: {k: [] for k in TOP_K} for s in styles}
    asked = 0

    for i, sample in enumerate(samples):
        entries, by_id = entries_for(sample)
        source = tmp / f"src{i}.jsonl"
        source.write_text("x", encoding="utf-8")
        index = MemorySearchIndex(tmp / f"db{i}.sqlite3", source)

        for qa in sample.get("qa", []):
            question = (qa.get("question") or "").strip()
            gold = {e for e in (qa.get("evidence") or []) if isinstance(e, str)}
            if not question or not gold:
                continue
            asked += 1
            words = keywords(question)

            plans = {
                "whole question": [question],
                # Each content word searched separately, results merged — the
                # closest this engine can come to an OR, since every query is
                # matched as one literal phrase.
                "keywords": words,
                "best keyword": [max(words, key=len)] if words else [],
            }
            for style, queries in plans.items():
                seen: list[str] = []
                for q in queries:
                    for hit in index.search(q, entries, limit=max(TOP_K)):
                        dia = by_id.get(hit.entry_id, "")
                        if dia and dia not in seen:
                            seen.append(dia)
                for k in TOP_K:
                    scores[style][k].append(1.0 if gold & set(seen[:k]) else 0.0)

    print(f"LoCoMo：{len(samples)} 段对话，{asked} 个问题，逐轮入库\n")
    print(f"{'查询方式':<16}" + "".join(f"{'recall@' + str(k):>12}" for k in TOP_K))
    print("-" * 52)
    for style in styles:
        row = "".join(f"{statistics.mean(scores[style][k]) * 100:>11.1f}%" for k in TOP_K)
        print(f"{style:<16}{row}")


if __name__ == "__main__":
    main()
