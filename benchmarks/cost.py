"""What each system costs to run, not how well it answers.

Accuracy has been measured to death and lands the field within a few points of
each other. Cost has not been measured at all, while being the thing this
project claims as its advantage — "no dependencies, and an index that survives
a write" is an assertion until somebody times the alternatives.

Only the systems that ingest without calling a model are timed here. The ones
that build a graph do their work in an LLM, so their wall-clock is a bill
rather than a benchmark and is reported separately from what was observed.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _extra in filter(None, os.environ.get("BENCH_SYS_PATH", "").split(":")):
    sys.path.append(_extra)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).with_name(os.environ.get("BENCH_DATA", "locomo10.json"))
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "2"))
QUERIES = 40
TOP_K = 8


def turns_and_queries():
    turns, queries = [], []
    for sample in json.loads(DATA.read_text(encoding="utf-8"))[:SAMPLES]:
        convo = sample["conversation"]
        names = sorted(
            (k for k in convo if k.startswith("session_") and "date" not in k),
            key=lambda n: int(n.split("_")[1]),
        )
        for name in names:
            when = convo.get(f"{name}_date_time", "")
            for t in convo[name]:
                if t.get("text"):
                    turns.append(f"[{when}] {t['speaker']}: {t['text']}")
        queries += [
            (qa.get("question") or "").strip()
            for qa in sample.get("qa", [])
            if (qa.get("question") or "").strip()
        ]
    return turns, queries[:QUERIES]


def tree_size(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return total / 1e6


def measure_ours(turns, queries):
    from kimi_cli.memory.entry import MemoryEntry
    from kimi_cli.memory.search import MemorySearchIndex

    entries = [MemoryEntry(kind="project", scope="persistent", content=t) for t in turns]
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "src"
    src.write_text("x" * len(turns), encoding="utf-8")
    index = MemorySearchIndex(tmp / "db", src)

    t0 = time.perf_counter()
    index.search("warmup", entries, limit=TOP_K)
    ingest = time.perf_counter() - t0

    times = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q, entries, limit=TOP_K)
        times.append((time.perf_counter() - t0) * 1000)

    # The claim being checked: a write does not cost a rebuild.
    entries.append(MemoryEntry(kind="project", scope="persistent", content="新增一条记忆"))
    src.write_text("x" * (len(turns) + 1), encoding="utf-8")
    t0 = time.perf_counter()
    index.search(queries[0], entries, limit=TOP_K)
    after_write = (time.perf_counter() - t0) * 1000

    return ingest, statistics.median(times), tree_size(tmp), after_write


def measure_txtai(turns, queries):
    from txtai import Embeddings

    emb = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True, hybrid=True)
    t0 = time.perf_counter()
    emb.index([(i, t, None) for i, t in enumerate(turns)])
    ingest = time.perf_counter() - t0
    times = []
    for q in queries:
        t0 = time.perf_counter()
        emb.search(q, TOP_K)
        times.append((time.perf_counter() - t0) * 1000)
    tmp = Path(tempfile.mkdtemp()) / "idx"
    emb.save(str(tmp))
    return ingest, statistics.median(times), tree_size(tmp), None


def measure_bm25(turns, queries):
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = [TextNode(text=t, id_=str(i)) for i, t in enumerate(turns)]
    t0 = time.perf_counter()
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)
    ingest = time.perf_counter() - t0
    times = []
    for q in queries:
        t0 = time.perf_counter()
        retriever.retrieve(q)
        times.append((time.perf_counter() - t0) * 1000)
    tmp = Path(tempfile.mkdtemp())
    retriever.persist(str(tmp))
    return ingest, statistics.median(times), tree_size(tmp), None


def measure_mem0(turns, queries):
    from mem0 import Memory

    store = Path(tempfile.mkdtemp())
    m = Memory.from_config(
        {
            "llm": {
                "provider": "deepseek",
                "config": {
                    "model": os.environ["BENCH_MODEL"],
                    "deepseek_base_url": os.environ["BENCH_BASE_URL"],
                    "api_key": os.environ["DEEPSEEK_API_KEY"],
                },
            },
            "embedder": {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5"}},
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "cost",
                    "path": str(store),
                    "on_disk": True,
                    "embedding_model_dims": 384,
                },
            },
        }
    )
    t0 = time.perf_counter()
    for t in turns:
        m.add([{"role": "user", "content": t}], user_id="c", infer=False)
    ingest = time.perf_counter() - t0
    times = []
    for q in queries:
        t0 = time.perf_counter()
        m.search(query=q, filters={"user_id": "c"}, top_k=TOP_K, threshold=0.0)
        times.append((time.perf_counter() - t0) * 1000)
    return ingest, statistics.median(times), tree_size(store), None


SYSTEMS = {
    "OpenKimo (SQLite FTS5)": measure_ours,
    "LlamaIndex BM25": measure_bm25,
    "txtai (稠密+BM25)": measure_txtai,
    "mem0 (qdrant, infer=False)": measure_mem0,
}


def main() -> None:
    turns, queries = turns_and_queries()
    print(f"{len(turns)} 轮原文，{len(queries)} 次查询\n")
    print(f"{'方案':<32}{'建库':>10}{'查询中位':>12}{'磁盘':>10}{'写后再查':>12}")
    print("-" * 78)
    for label, fn in SYSTEMS.items():
        try:
            ingest, q_ms, disk, after = fn(turns, queries)
        except Exception as exc:
            print(f"{label:<30}  失败: {type(exc).__name__}: {str(exc)[:40]}")
            continue
        tail = f"{after:>10.1f}ms" if after is not None else f"{'—':>12}"
        print(f"{label:<30}{ingest:>9.1f}s{q_ms:>10.1f}ms{disk:>9.1f}MB{tail}")


if __name__ == "__main__":
    main()
