"""FTS5 against a dense vector store, on identical stored content.

The earlier run compared whole systems and was worthless: ours stored verbatim
turns and mem0 stored LLM-rewritten facts, while the metric was "does the gold
answer string appear in what came back". Verbatim storage wins that by
construction. Two of the four samples also had mem0's own extraction fail, and
chroma silently disabled mem0's hybrid scoring.

So this isolates the one thing both systems genuinely have in common — the
retriever. Identical turns go into both (`infer=False` keeps mem0 from
rewriting them), and scoring uses LoCoMo's own evidence labels rather than
string containment, so no storage format is flattered. What this does NOT
measure is mem0's extraction layer; that is a separate axis and the earlier run
is not evidence about it either way.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# mem0 and its embedder are not dependencies of this project; point BENCH_SYS_PATH
# at wherever they were installed, or leave it unset if they are importable.
for _extra in filter(None, os.environ.get("BENCH_SYS_PATH", "").split(":")):
    sys.path.insert(0, _extra)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).with_name(os.environ.get("BENCH_DATA", "locomo10.json"))
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "4"))
#: bge-small is English-only. On Chinese it is not a weak retriever, it is the
#: wrong tool, so a Chinese run must name a multilingual model or it measures
#: nothing but the language mismatch.
EMBED_MODEL = os.environ.get("BENCH_EMBED", "BAAI/bge-small-en-v1.5")
EMBED_DIMS = int(os.environ.get("BENCH_EMBED_DIMS", "384"))
TOP_K = (1, 3, 8)


def turns_of(sample: dict) -> list[tuple[str, str]]:
    """(dia_id, "speaker: text") for every turn, in order."""
    convo = sample["conversation"]
    names = sorted(
        (k for k in convo if k.startswith("session_") and "date" not in k),
        key=lambda n: int(n.split("_")[1]),
    )
    out = []
    for name in names:
        for turn in convo[name]:
            if turn.get("text") and turn.get("dia_id"):
                out.append((turn["dia_id"], f"{turn['speaker']}: {turn['text']}"))
    return out


def questions_of(sample: dict) -> list[tuple[str, set[str]]]:
    """Questions that carry gold evidence ids; the rest cannot be scored."""
    out = []
    for qa in sample.get("qa", []):
        question = (qa.get("question") or "").strip()
        evidence = qa.get("evidence") or []
        if question and evidence:
            out.append((question, {str(e) for e in evidence}))
    return out


def score(ranked: list[str], gold: set[str]) -> dict[int, float]:
    return {k: 1.0 if gold & set(ranked[:k]) else 0.0 for k in TOP_K}


def run_ours(turns, questions):
    from kimi_cli.memory.entry import MemoryEntry
    from kimi_cli.memory.search import MemorySearchIndex

    entries = [MemoryEntry(kind="project", scope="persistent", content=text) for _, text in turns]
    dia_by_entry = {e.id: dia for e, (dia, _) in zip(entries, turns, strict=True)}

    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").write_text("x", encoding="utf-8")
    index = MemorySearchIndex(tmp / "db", tmp / "src")

    out = []
    for question, gold in questions:
        hits = index.search(question, entries, limit=max(TOP_K))
        out.append(score([dia_by_entry[h.entry_id] for h in hits], gold))
    return out


def _chroma(tag, path):
    return {
        "provider": "chroma",
        "config": {"collection_name": tag, "path": str(path)},
    }


def _qdrant(tag, path):
    """mem0's own hybrid mode — dense plus BM25 — which chroma cannot do.

    Without this the comparison is lexical-vs-dense and quietly withholds from
    mem0 the half of its scoring that most resembles ours.
    """
    return {
        "provider": "qdrant",
        "config": {
            "collection_name": tag,
            "path": str(path),
            "on_disk": True,
            # bge-small is 384-dim; the default collection is built at 1536.
            "embedding_model_dims": EMBED_DIMS,
        },
    }


def run_mem0(turns, questions, tag, store_cfg):
    from mem0 import Memory

    store = Path(tempfile.mkdtemp())
    memory = Memory.from_config(
        {
            # Never called — `infer=False` skips extraction — but mem0 builds
            # the client eagerly, so it still has to be a valid provider.
            "llm": {
                "provider": "deepseek",
                "config": {
                    "model": os.environ["BENCH_MODEL"],
                    "deepseek_base_url": os.environ["BENCH_BASE_URL"],
                    "api_key": os.environ["DEEPSEEK_API_KEY"],
                },
            },
            "embedder": {"provider": "fastembed", "config": {"model": EMBED_MODEL}},
            "vector_store": store_cfg(tag, store),
        }
    )
    for dia_id, text in turns:
        memory.add(
            [{"role": "user", "content": text}],
            user_id=tag,
            infer=False,  # store the turn as given; no LLM rewrite
            metadata={"dia_id": dia_id},
        )

    out = []
    for question, gold in questions:
        # `limit=` is silently swallowed by **kwargs — the parameter is `top_k`.
        # `threshold` defaults to 0.1, which on Chinese cuts almost everything:
        # the correct hit scored 0.101. Scoring recall against a relevance
        # filter measures the filter, so it is disabled here.
        found = memory.search(
            query=question,
            filters={"user_id": tag},
            top_k=max(TOP_K),
            threshold=0.0,
        )
        rows = found.get("results", found) if isinstance(found, dict) else found
        ranked = [str((r.get("metadata") or {}).get("dia_id", "")) for r in rows]
        out.append(score(ranked, gold))
    return out


def main() -> None:
    with DATA.open(encoding="utf-8") as handle:
        samples = json.load(handle)[:SAMPLES]

    ours, theirs, hybrid, asked, stored = [], [], [], 0, 0
    for i, sample in enumerate(samples):
        turns = turns_of(sample)
        questions = questions_of(sample)
        asked += len(questions)
        stored += len(turns)
        ours += run_ours(turns, questions)
        theirs += run_mem0(turns, questions, f"fair{i}", _chroma)
        hybrid += run_mem0(turns, questions, f"hyb{i}", _qdrant)
        print(
            f"  样本 {i + 1}/{len(samples)} 完成（{len(turns)} 轮，{len(questions)} 问）",
            flush=True,
        )

    print(f"\n{len(samples)} 段对话，{stored} 轮原文，{asked} 个带证据标注的问题")
    print("指标：标准证据轮是否进入 top-k（与存储形式无关）\n")
    header = "".join(f"{'recall@' + str(k):>11}" for k in TOP_K)
    print(f"{'检索器':<34}{header}")
    print("-" * (34 + 11 * len(TOP_K)))
    for label, rows in (
        ("OpenKimo (SQLite FTS5 词法)", ours),
        (f"mem0 (纯向量 {EMBED_MODEL.split('/')[-1]})", theirs),
        ("mem0 (qdrant 混合: 向量+BM25)", hybrid),
    ):
        cells = "".join(f"{statistics.mean(r[k] for r in rows) * 100:>10.1f}%" for k in TOP_K)
        print(f"{label:<32}{cells}")


if __name__ == "__main__":
    main()
