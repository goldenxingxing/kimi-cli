"""Does a good BM25 with Chinese handling match six hundred lines of ours?

The previous attempt passed a tokenizer to LlamaIndex's retriever, which warns
that the parameter is deprecated and then discards it — the run produced digits
identical to the default, which would have read as "a Chinese tokenizer does
not help" and happened to favour us. Writing BM25 by hand instead avoided that
but understated BM25: a bare `split()` scores 40.1% in English where
LlamaIndex's stemming implementation scores 60.6%.

So this keeps LlamaIndex's implementation and does the tokenizing upstream of
it, by rewriting Chinese runs as space-separated bigrams in both the corpus and
the query. Its default tokenizer then sees words it can handle, and the
comparison is our retrieval against a mature BM25 that has been told how
Chinese works.

The question being settled is whether most of our retrieval code can be
deleted. It buys two things that are not in doubt — no dependencies beyond the
standard library, and an index that survives a write without being rebuilt —
and the accuracy it buys is what this measures.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TOP_K = (1, 3, 8)
DATA = Path(__file__).with_name(os.environ.get("BENCH_DATA", "locomo10.json"))
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "2"))

_CJK_RUN = re.compile(r"[一-鿿]+")


def bigrammed(text: str) -> str:
    """Chinese runs rewritten as space-separated two-character windows.

    Two characters is the commonest Chinese word length, and a whitespace
    tokenizer cannot find it on its own — which is the whole of the 0% BM25
    scores, not anything about BM25 itself.
    """

    def expand(match: re.Match[str]) -> str:
        run = match.group(0)
        if len(run) == 1:
            return f" {run} "
        return " " + " ".join(run[i : i + 2] for i in range(len(run) - 1)) + " "

    return _CJK_RUN.sub(expand, text)


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


def ours(turns, questions):
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


def llama_bm25(turns, questions, *, prepare=lambda s: s):
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = [TextNode(text=prepare(t), id_=str(i)) for i, (_, t) in enumerate(turns)]
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=max(TOP_K))
    out = []
    for q, g in questions:
        got = retriever.retrieve(prepare(q))
        out.append(score([turns[int(n.node.id_)][0] for n in got], g))
    return out


def main():
    data = list(corpus())
    systems = {
        "OpenKimo（653 行，零依赖）": ours,
        "LlamaIndex BM25（原样）": lambda t, q: llama_bm25(t, q),
        "LlamaIndex BM25 + 中文双字预切": lambda t, q: llama_bm25(t, q, prepare=bigrammed),
    }
    print(f"{DATA.name}：{sum(len(q) for _, q in data)} 题\n")
    print(f"{'方案':<36}" + "".join(f"{'recall@' + str(k):>11}" for k in TOP_K))
    print("-" * (34 + 11 * len(TOP_K)))
    for label, fn in systems.items():
        rows = []
        for turns, questions in data:
            rows += fn(turns, questions)
        print(
            f"{label:<34}"
            + "".join(f"{statistics.mean(r[k] for r in rows) * 100:>10.1f}%" for k in TOP_K)
        )


if __name__ == "__main__":
    main()
