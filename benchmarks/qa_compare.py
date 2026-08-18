"""The same systems, scored the way a user would score them.

Retrieval recall and answer accuracy are different questions, and a system can
be good at one and bad at the other — a retriever that returns the right turn
buried in seven wrong ones has perfect recall@8 and gives the model a worse
prompt than one that returns three tight ones. Running both metrics over the
same systems is what makes the graph numbers, added later, mean anything.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Appended, not prepended — see the note in final_qa.py. This copy was missed
# when that one was fixed, and importing this module was enough to reinstate
# the problem, which is what a second copy of a path hack is for.
for _extra in filter(None, os.environ.get("BENCH_SYS_PATH", "").split(":")):
    sys.path.append(_extra)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge import score_all  # noqa: E402  (must follow the sys.path setup above)
from oss_compare import DATA, SAMPLES  # noqa: E402  (must follow the sys.path setup above)

TOP_K = 8


def dated_corpus():
    """Turns carrying the date of the session they were said in.

    Nearly a quarter of LoCoMo's answers are dates, and the date lives in the
    conversation's `session_N_date_time` field rather than in any turn — so a
    system that retrieves exactly the right turn still cannot answer "when did
    Caroline go to the support group". Scoring that as a failure measures the
    harness. Every system gets the same dated text, so this changes the ceiling
    rather than the ranking.
    """
    for sample in json.loads(DATA.read_text(encoding="utf-8"))[:SAMPLES]:
        convo = sample["conversation"]
        names = sorted(
            (k for k in convo if k.startswith("session_") and "date" not in k),
            key=lambda n: int(n.split("_")[1]),
        )
        turns = []
        for name in names:
            when = convo.get(f"{name}_date_time", "")
            for turn in convo[name]:
                if turn.get("text") and turn.get("dia_id"):
                    stamp = f"[{when}] " if when else ""
                    turns.append((turn["dia_id"], f"{stamp}{turn['speaker']}: {turn['text']}"))
        yield turns, []


def ours(turns, questions):
    from kimi_cli.memory.entry import MemoryEntry
    from kimi_cli.memory.search import MemorySearchIndex

    entries = [MemoryEntry(kind="project", scope="persistent", content=t) for _, t in turns]
    text = {e.id: e.content for e in entries}
    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").write_text("x", encoding="utf-8")
    index = MemorySearchIndex(tmp / "db", tmp / "src")
    return [
        "\n".join(text[h.entry_id] for h in index.search(q, entries, limit=TOP_K))
        for q, _ in questions
    ]


def txtai_hybrid(turns, questions):
    from txtai import Embeddings

    emb = Embeddings(
        path=os.environ.get("BENCH_TXTAI_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        content=True,
        hybrid=True,
    )
    emb.index([(i, text, None) for i, (_, text) in enumerate(turns)])
    return ["\n".join(turns[int(r["id"])][1] for r in emb.search(q, TOP_K)) for q, _ in questions]


def llamaindex_bm25(turns, questions):
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = [TextNode(text=text, id_=str(i)) for i, (_, text) in enumerate(turns)]
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)
    return ["\n".join(n.node.get_content() for n in retriever.retrieve(q)) for q, _ in questions]


SYSTEMS = {
    "OpenKimo (FTS5 + IDF scan)": ours,
    "txtai (稠密+BM25 混合)": txtai_hybrid,
    "LlamaIndex BM25": llamaindex_bm25,
}


def gold_answers(limit_samples: int):
    """Questions paired with their written answer, not their evidence ids."""
    out = []
    for sample in json.loads(DATA.read_text(encoding="utf-8"))[:limit_samples]:
        rows = []
        for qa in sample.get("qa", []):
            q = (qa.get("question") or "").strip()
            a = str(qa.get("answer") or "").strip()
            if q and a:
                rows.append((q, a))
        out.append(rows)
    return out


async def main() -> None:
    data = list(dated_corpus())
    golds = gold_answers(SAMPLES)
    cap = int(os.environ.get("BENCH_QUESTIONS", "120"))

    retrieved: dict[str, list[tuple[str, str, str]]] = {}
    for label, fn in SYSTEMS.items():
        rows: list[tuple[str, str, str]] = []
        for (turns, _), qa in zip(data, golds, strict=True):
            picked = qa[: max(1, cap // len(data))]
            notes = fn(turns, [(q, "") for q, _ in picked])
            rows += [(q, a, n) for (q, a), n in zip(picked, notes, strict=True)]
        retrieved[label] = rows
        print(f"  {label}: 取回 {len(rows)} 题", flush=True)

    print("\n判分中…", flush=True)
    judged = await score_all(retrieved)

    print(f"\n{DATA.name}，{len(next(iter(judged.values())))} 题，LLM 判分的端到端问答准确率\n")
    print(f"{'方案':<34}{'准确率':>10}")
    print("-" * 46)
    for label, rows in judged.items():
        print(f"{label:<32}{statistics.mean(1.0 if r.correct else 0.0 for r in rows) * 100:>9.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
