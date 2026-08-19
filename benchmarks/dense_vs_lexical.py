"""What semantic retrieval would buy, on the turns it would run against.

Auto-injecting on every turn only works if what gets injected is usually
relevant, and lexical retrieval reaches 17.5% useful on real turns. A coverage
threshold cannot lift that — the misses are semantic, not lexical — so the
question becomes whether embeddings can, and at what size.

Same store, same turns, same judge; the only thing that changes is the
retriever. Both are also timed, because the cost side of this decision is
first-index and per-query latency as much as megabytes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.search import MemorySearchIndex

TOP_K = 3
MODEL = os.environ.get("DENSE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

_USEFUL = """\
下面是一个人刚说的一句话，以及检索系统为这句话取回的几条长期记忆。

对每一条判断它是否会实质帮助理解或执行这句话——提供了这句话本身没有、而助手又
需要的信息。仅仅主题沾边不算，宁可判否。

按顺序只回同样数量的 Y 或 N，逗号分隔，比如 `N,Y,N`。

他说的话：{turn}

取回的记忆：
{hits}"""


async def ask(client, sem, prompt: str) -> str:
    async with sem:
        for _ in range(3):
            try:
                r = await client.post(
                    os.environ["BENCH_BASE_URL"].rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                    json={
                        "model": os.environ["BENCH_MODEL"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                    },
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                await asyncio.sleep(1.2)
    return ""


def lexical(store: list[str], turns: list[str]):
    entries = [MemoryEntry(kind="project", scope="persistent", content=f) for f in store]
    text = {e.id: e.content for e in entries}
    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").write_text("x", encoding="utf-8")
    index = MemorySearchIndex(tmp / "db", tmp / "src")

    started = time.perf_counter()
    index.search("warmup", entries, limit=TOP_K)
    build = time.perf_counter() - started

    started = time.perf_counter()
    out = [[text[h.entry_id] for h in index.search(t, entries, limit=TOP_K)] for t in turns]
    per_query = (time.perf_counter() - started) / max(1, len(turns)) * 1000
    return out, build, per_query


def dense(store: list[str], turns: list[str]):
    import numpy as np
    from fastembed import TextEmbedding

    started = time.perf_counter()
    model = TextEmbedding(MODEL)
    vectors = np.array([v for v in model.embed(store)])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    build = time.perf_counter() - started

    started = time.perf_counter()
    out = []
    for t in turns:
        q = next(iter(model.embed([t])))
        q = q / (np.linalg.norm(q) + 1e-9)
        order = (vectors @ q).argsort()[::-1][:TOP_K]
        out.append([store[i] for i in order])
    per_query = (time.perf_counter() - started) / max(1, len(turns)) * 1000
    return out, build, per_query


async def judge(client, sem, turns, retrieved):
    """Returns `(useful, injected, per_position)`.

    Position matters more than the total. If only the first result is ever
    useful, the fix for auto-injection is to inject one, not to buy a better
    retriever — and a rate averaged over three hides that completely.
    """
    replies = await asyncio.gather(
        *(
            ask(
                client,
                sem,
                _USEFUL.format(turn=t, hits="\n".join(f"{i + 1}. {h}" for i, h in enumerate(hits))),
            )
            if hits
            else asyncio.sleep(0, result="")
            for t, hits in zip(turns, retrieved, strict=True)
        )
    )
    total = good = 0
    at = [[0, 0] for _ in range(TOP_K)]  # [useful, shown] per position
    for hits, reply in zip(retrieved, replies, strict=True):
        marks = [m.upper() for m in re.findall(r"[YN]", reply or "")][: len(hits)]
        total += len(hits)
        good += sum(1 for m in marks if m == "Y")
        for i, mark in enumerate(marks):
            if i < TOP_K:
                at[i][1] += 1
                at[i][0] += mark == "Y"
    return good, total, at


async def main() -> None:
    store = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    turns = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    print(f"记忆库 {len(store)} 条 · 真实发言 {len(turns)} 条 · 每轮 top-{TOP_K}\n")

    lex, lex_build, lex_q = lexical(store, turns)
    den, den_build, den_q = dense(store, turns)

    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=180) as client:
        lex_good, lex_total, lex_at = await judge(client, sem, turns, lex)
        den_good, den_total, den_at = await judge(client, sem, turns, den)

    print(
        f"{'检索器':<26}{'注入条数':>10}{'其中有用':>10}{'有用率':>10}{'建库':>11}{'每次查询':>11}"
    )
    print("-" * 80)
    print(
        f"{'词法（现状）':<24}{lex_total:>10}{lex_good:>10}"
        f"{lex_good / max(1, lex_total) * 100:>9.1f}%{lex_build * 1000:>9.0f}ms{lex_q:>9.1f}ms"
    )
    print(
        f"{'稠密向量（多语言）':<22}{den_total:>10}{den_good:>10}"
        f"{den_good / max(1, den_total) * 100:>9.1f}%{den_build * 1000:>9.0f}ms{den_q:>9.1f}ms"
    )

    print(
        f"\n{'按位置的有用率':<26}"
        + "".join(f"{'第' + str(i + 1) + '条':>10}" for i in range(TOP_K))
    )
    print("-" * 58)
    for label, at in (("词法", lex_at), ("稠密向量", den_at)):
        cells = "".join(f"{(u / s * 100 if s else 0):>9.1f}%" for u, s in at)
        print(f"{label:<24}{cells}")

    both = sum(1 for a, b in zip(lex, den, strict=True) if set(a) & set(b))
    print(f"\n两者取回结果有重叠的轮次：{both} / {len(turns)}")


if __name__ == "__main__":
    asyncio.run(main())
