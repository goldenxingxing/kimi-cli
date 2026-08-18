"""Every system that could be run, on one corpus, under one judge.

Retrieval recall and answer accuracy disagree — a system can put the right turn
in the top eight more often and still produce worse answers, because the other
seven dilute it — so both are reported rather than one standing in for the
other.

Systems that answer for themselves (Cognee's graph path) are marked as such:
their answer is judged directly, so the answering model is theirs rather than
the shared one. That is a real asymmetry and is labelled, not averaged away.
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
# Appended, not prepended. These directories are `pip install --target` trees
# for one library and carry its whole dependency closure, so putting them first
# lets them decide versions for everything else — here a tokenizers 0.23.1 that
# the venv's transformers refuses, which surfaced as "Transformers is not
# installed" and silently dropped a system from the comparison.
for _extra in filter(None, os.environ.get("BENCH_SYS_PATH", "").split(":")):
    sys.path.append(_extra)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402  (must follow the sys.path setup above)
from judge import Judged, Model, score_all  # noqa: E402  (must follow the sys.path setup above)
from qa_compare import (  # noqa: E402  (must follow the sys.path setup above)
    dated_corpus,
    gold_answers,
)

TOP_K = 8
DATA_NAME = os.environ.get("BENCH_DATA", "locomo10.json")
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "2"))
PER_SAMPLE = int(os.environ.get("BENCH_PER_SAMPLE", "60"))


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
        for q in questions
    ]


def txtai_hybrid(turns, questions):
    from txtai import Embeddings

    emb = Embeddings(
        path=os.environ.get("BENCH_TXTAI_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        content=True,
        hybrid=True,
    )
    emb.index([(i, t, None) for i, (_, t) in enumerate(turns)])
    return ["\n".join(turns[int(r["id"])][1] for r in emb.search(q, TOP_K)) for q in questions]


def llamaindex_bm25(turns, questions):
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    nodes = [TextNode(text=t, id_=str(i)) for i, (_, t) in enumerate(turns)]
    r = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)
    return ["\n".join(n.node.get_content() for n in r.retrieve(q)) for q in questions]


def mem0_hybrid(turns, questions, tag="final"):
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
                    "collection_name": f"q_{tag}",
                    "path": str(store),
                    "on_disk": True,
                    "embedding_model_dims": 384,
                },
            },
        }
    )
    for _, text in turns:
        m.add([{"role": "user", "content": text}], user_id=tag, infer=False)
    out = []
    for q in questions:
        found = m.search(query=q, filters={"user_id": tag}, top_k=TOP_K, threshold=0.0)
        rows = found.get("results", found) if isinstance(found, dict) else found
        out.append("\n".join(str(r.get("memory", "")) for r in rows[:TOP_K]))
    return out


RETRIEVERS = {
    "OpenKimo (FTS5 + IDF scan)": ours,
    "txtai (稠密+BM25 混合)": txtai_hybrid,
    "LlamaIndex BM25": llamaindex_bm25,
    "mem0 (qdrant 混合)": mem0_hybrid,
}


async def main() -> None:
    data = list(dated_corpus())
    golds = gold_answers(SAMPLES)

    retrieved: dict[str, list[tuple[str, str, str]]] = {}
    for label, fn in RETRIEVERS.items():
        rows = []
        try:
            for si, ((turns, _), qa) in enumerate(zip(data, golds, strict=True)):
                picked = qa[:PER_SAMPLE]
                notes = (
                    fn(turns, [q for q, _ in picked])
                    if fn is not mem0_hybrid
                    else fn(turns, [q for q, _ in picked], tag=f"s{si}")
                )
                rows += [(q, a, n) for (q, a), n in zip(picked, notes, strict=True)]
        except Exception as exc:
            print(f"  {label} 跑失败: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        retrieved[label] = rows
        print(f"  {label}: {len(rows)} 题", flush=True)

    # Cognee's two paths come from a file: it needs its own environment.
    cognee_path = Path(os.environ.get("BENCH_COGNEE", "/dev/null"))
    direct: dict[str, list[tuple[str, str, str]]] = {}
    if cognee_path.exists():
        rows = json.loads(cognee_path.read_text(encoding="utf-8"))
        retrieved["Cognee (取回原文块)"] = [
            (r["question"], r["gold"], r.get("notes", "")) for r in rows
        ]
        direct["Cognee (图谱自答)"] = [
            (r["question"], r["gold"], r.get("graph_answer", "")) for r in rows
        ]
        print(f"  Cognee: {len(rows)} 题（两条路径）", flush=True)

    print("\n判分中…", flush=True)
    judged = await score_all(retrieved)

    # Control. Cognee answers with its own model and everyone else is answered
    # by ours, so part of any gap is the model rather than the memory. Running
    # one retriever's notes through Cognee's model separates the two; without
    # it the comparison cannot say which was being measured.
    control_key = "OpenKimo (FTS5 + IDF scan)"
    if direct and control_key in retrieved:
        control = await score_all(
            {f"{control_key} — 换 k3 作答（对照）": retrieved[control_key]}, alt=True
        )
        judged.update(control)

    if direct:
        model = Model(8)
        async with httpx.AsyncClient(timeout=180) as client:
            import re as _re

            from judge import _JUDGE_PROMPT

            for label, rows in direct.items():

                async def one(q, gold, given):
                    if not given.strip():
                        return Judged(q, gold, "(空)", correct=False)
                    v = await model.ask(
                        client, _JUDGE_PROMPT.format(question=q, gold=gold, given=given)
                    )
                    return Judged(q, gold, given, correct=bool(_re.match(r"\s*CORRECT", v, _re.I)))

                judged[label] = list(await asyncio.gather(*(one(q, g, a) for q, g, a in rows)))

    print(f"\n{DATA_NAME}，LLM 判分的端到端问答准确率\n")
    print(f"{'方案':<34}{'题数':>7}{'准确率':>10}")
    print("-" * 52)
    for label, rows in sorted(
        judged.items(), key=lambda kv: -statistics.mean(1.0 if r.correct else 0.0 for r in kv[1])
    ):
        acc = statistics.mean(1.0 if r.correct else 0.0 for r in rows) * 100
        print(f"{label:<32}{len(rows):>7}{acc:>9.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
