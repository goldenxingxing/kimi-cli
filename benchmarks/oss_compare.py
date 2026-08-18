"""Several open-source retrievers, one corpus, one metric.

Adding systems is not just scorekeeping: every comparison so far has turned up
something wrong in our own code — extraction that silently produced nothing,
a scan that weighted grammar as heavily as content, a frequency cutoff that
emptied small stores. A second and third implementation of the same idea is the
cheapest way to find the next one.

Ground rules, same as the mem0 run: identical text goes into every system, and
scoring uses LoCoMo's own evidence turn ids, so no storage format is flattered
by the metric. Systems whose value is in synthesis rather than retrieval — the
graph ones — are handicapped by this and are reported separately if run at all.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

TOP_K = (1, 3, 8)
HERE = Path(__file__).parent
DATA = HERE / os.environ.get("BENCH_DATA", "locomo10.json")
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "2"))


def corpus():
    for sample in json.loads(DATA.read_text(encoding="utf-8"))[:SAMPLES]:
        convo = sample["conversation"]
        names = sorted(
            (k for k in convo if k.startswith("session_") and "date" not in k),
            key=lambda n: int(n.split("_")[1]),
        )
        turns = [
            (t["dia_id"], f"{t['speaker']}: {t['text']}")
            for n in names
            for t in convo[n]
            if t.get("text") and t.get("dia_id")
        ]
        qs = [
            (qa["question"].strip(), {str(e) for e in qa["evidence"]})
            for qa in sample.get("qa", [])
            if (qa.get("question") or "").strip() and qa.get("evidence")
        ]
        yield turns, qs


def score(ranked, gold):
    return {k: 1.0 if gold & set(ranked[:k]) else 0.0 for k in TOP_K}


# ------------------------------------------------------------------ systems


def ours(turns, questions):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import tempfile

    from kimi_cli.memory.entry import MemoryEntry
    from kimi_cli.memory.search import MemorySearchIndex

    entries = [MemoryEntry(kind="project", scope="persistent", content=t) for _, t in turns]
    dia = {e.id: d for e, (d, _) in zip(entries, turns, strict=True)}
    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").write_text("x", encoding="utf-8")
    index = MemorySearchIndex(tmp / "db", tmp / "src")
    return [
        score([dia[h.entry_id] for h in index.search(q, entries, limit=max(TOP_K))], g)
        for q, g in questions
    ]


def txtai_hybrid(turns, questions):
    """txtai's own hybrid: dense embeddings plus its BM25 index."""
    from txtai import Embeddings

    emb = Embeddings(
        path=os.environ.get("BENCH_TXTAI_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        content=True,
        hybrid=True,
    )
    emb.index([(i, text, None) for i, (_, text) in enumerate(turns)])
    out = []
    for q, g in questions:
        rows = emb.search(q, max(TOP_K))
        out.append(score([turns[int(r["id"])][0] for r in rows], g))
    return out


def llamaindex_bm25(turns, questions):
    """A conventional BM25 with a real tokenizer, as a lexical reference point."""
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = [TextNode(text=text, id_=str(i)) for i, (_, text) in enumerate(turns)]
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=max(TOP_K))
    out = []
    for q, g in questions:
        got = retriever.retrieve(q)
        out.append(score([turns[int(n.node.id_)][0] for n in got], g))
    return out


SYSTEMS = {
    "OpenKimo (FTS5 + IDF scan)": ours,
    "txtai (稠密+BM25 混合)": txtai_hybrid,
    "LlamaIndex BM25": llamaindex_bm25,
}


def main():
    only = set(sys.argv[1:])
    data = list(corpus())
    n_q = sum(len(q) for _, q in data)
    print(f"{DATA.name}：{sum(len(t) for t, _ in data)} 轮，{n_q} 问\n")
    print(f"{'方案':<34}" + "".join(f"{'recall@' + str(k):>11}" for k in TOP_K))
    print("-" * (32 + 11 * len(TOP_K)))
    for label, fn in SYSTEMS.items():
        if only and label.split()[0] not in only:
            continue
        rows = []
        try:
            for turns, questions in data:
                rows += fn(turns, questions)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<32}  跑失败: {type(exc).__name__}: {exc}"[:110])
            continue
        print(
            f"{label:<32}"
            + "".join(f"{statistics.mean(r[k] for r in rows) * 100:>10.1f}%" for k in TOP_K)
        )


if __name__ == "__main__":
    main()
