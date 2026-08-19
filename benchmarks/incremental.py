"""Does memory stay current when the facts change?

Every benchmark here loads a fixed corpus and then asks about it, which tests
retrieval and cannot test maintenance — and maintenance is where this project's
design sits: supersession, dating, retirement, approval. It is also where mem0
made the opposite bet, abandoning UPDATE/DELETE consolidation in 2026-04 for
single-pass accumulation. Neither bet has data.

So sessions arrive in order, six of the nine facts are revised partway through,
and the questions are asked at the end about the *current* value. Three answers
are possible and they mean different things:

- **current** — maintenance worked
- **stale** — the superseded value survived and won
- **both** — nothing was lost, but the reader is handed a contradiction

Each system writes through its own real path: this one extracts and upserts,
with the dedup and merge it ships; mem0 uses `infer=True`, which is where its
accumulation decision lives. Approval is bypassed here — what is under test is
the maintenance mechanism, not the consent step that guards it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _extra in filter(None, os.environ.get("BENCH_SYS_PATH", "").split(":")):
    sys.path.append(_extra)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
from judge import Model

CORPUS = Path(__file__).with_name("incremental_corpus.json")
TOP_K = 5

_VERDICT = """\
下面是一条记忆系统为某个问题取回的内容，以及这条事实的**当前值**与**已被取代的旧值**。

判断取回的内容支持哪一种，只回一个词：

CURRENT  — 只支持当前值
STALE    — 只支持旧值
BOTH     — 两个值都出现了，且没有指明哪个是现行的
NEITHER  — 两个都没有

问题：{q}
当前值：{current}
旧值：{stale}

取回的内容：
{notes}"""

_VERDICT_KEPT = """\
下面是一条记忆系统为某个问题取回的内容，以及正确答案。

取回的内容是否支持这个答案？只回 YES 或 NO。

问题：{q}
正确答案：{current}

取回的内容：
{notes}"""


def transcript(session: dict) -> str:
    head = f"[{session['date']}]\n"
    return head + "\n".join(f"{who}: {text}" for who, text in session["turns"])


async def run_ours(model: Model, client: httpx.AsyncClient, sessions, questions):
    """Extraction and upsert — the write path this project actually ships."""
    from kimi_cli.memory.archivist import _EXTRACTION_PROMPT, _parse_candidates
    from kimi_cli.memory.entry import MemoryEntry
    from kimi_cli.memory.search import MemorySearchIndex
    from kimi_cli.memory.storage import read_entries, upsert_entry

    store = Path(tempfile.mkdtemp()) / "persistent.jsonl"
    for session in sessions:
        reply = await model.ask(
            client,
            _EXTRACTION_PROMPT.format(conversation=transcript(session), today=session["date"]),
        )
        for candidate in _parse_candidates(reply, session_id=str(session["id"])):
            upsert_entry(
                store,
                MemoryEntry(
                    kind=candidate.kind,
                    scope="persistent",
                    content=candidate.content,
                    key=candidate.key,
                ),
            )

    entries = read_entries(store)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").write_text("x", encoding="utf-8")
    index = MemorySearchIndex(tmp / "db", tmp / "src")
    text = {e.id: e.content for e in entries}
    notes = [
        "\n".join(text[h.entry_id] for h in index.search(q, entries, limit=TOP_K))
        for q in questions
    ]
    return notes, len(entries)


async def run_mem0(model: Model, client: httpx.AsyncClient, sessions, questions):
    from mem0 import Memory

    path = Path(tempfile.mkdtemp())
    memory = Memory.from_config(
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
                    "collection_name": "incr",
                    "path": str(path),
                    "on_disk": True,
                    "embedding_model_dims": 384,
                },
            },
        }
    )
    for session in sessions:
        memory.add(
            [{"role": who, "content": text} for who, text in session["turns"]],
            user_id="u",
            infer=True,
        )
    notes = []
    for q in questions:
        found = memory.search(query=q, filters={"user_id": "u"}, top_k=TOP_K, threshold=0.0)
        rows = found.get("results", found) if isinstance(found, dict) else found
        notes.append("\n".join(str(r.get("memory", "")) for r in rows[:TOP_K]))
    stored = memory.get_all(filters={"user_id": "u"}, limit=500)
    rows = stored.get("results", stored) if isinstance(stored, dict) else stored
    return notes, len(rows)


async def classify(model, client, facts, notes):
    async def one(fact, note):
        if fact["stale"] is None:
            reply = await model.ask(
                client,
                _VERDICT_KEPT.format(q=fact["q"], current=fact["current"], notes=note or "(空)"),
            )
            return "KEPT" if re.match(r"\s*YES", reply or "", re.I) else "LOST"
        reply = await model.ask(
            client,
            _VERDICT.format(
                q=fact["q"], current=fact["current"], stale=fact["stale"], notes=note or "(空)"
            ),
        )
        match = re.search(r"\b(CURRENT|STALE|BOTH|NEITHER)\b", reply or "", re.I)
        return match.group(1).upper() if match else "NEITHER"

    return list(await asyncio.gather(*(one(f, n) for f, n in zip(facts, notes, strict=True))))


async def main() -> None:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    sessions, facts = data["sessions"], data["facts"]
    questions = [f["q"] for f in facts]
    changed = [f for f in facts if f["stale"] is not None]
    kept = [f for f in facts if f["stale"] is None]

    print(
        f"{len(sessions)} 个会话按时间顺序写入 · {len(changed)} 条事实发生过变更 · "
        f"{len(kept)} 条从未变更\n"
    )

    model = Model(6)
    runners = {"OpenKimo（抽取+去重合并）": run_ours}
    if os.environ.get("BENCH_SYS_PATH"):
        runners["mem0（infer=True，单遍累积）"] = run_mem0

    async with httpx.AsyncClient(timeout=240) as client:
        results = {}
        for label, runner in runners.items():
            try:
                notes, stored = await runner(model, client, sessions, questions)
            except Exception as exc:
                print(f"  {label} 跑失败: {type(exc).__name__}: {str(exc)[:80]}")
                continue
            verdicts = await classify(model, client, facts, notes)
            results[label] = (verdicts, stored)
            print(f"  {label}: 库里 {stored} 条", flush=True)

    print(f"\n{'方案':<30}{'当前值':>8}{'旧值':>7}{'两者':>7}{'都没有':>8}{'留存':>9}")
    print("-" * 74)
    for label, (verdicts, _) in results.items():
        c = sum(1 for f, v in zip(facts, verdicts, strict=True) if f["stale"] and v == "CURRENT")
        s = sum(1 for f, v in zip(facts, verdicts, strict=True) if f["stale"] and v == "STALE")
        b = sum(1 for f, v in zip(facts, verdicts, strict=True) if f["stale"] and v == "BOTH")
        n = sum(1 for f, v in zip(facts, verdicts, strict=True) if f["stale"] and v == "NEITHER")
        k = sum(1 for f, v in zip(facts, verdicts, strict=True) if not f["stale"] and v == "KEPT")
        print(f"{label:<28}{c:>7}/{len(changed)}{s:>7}{b:>7}{n:>8}{k:>7}/{len(kept)}")

    print("\n逐条：")
    for i, fact in enumerate(facts):
        marks = "  ".join(f"{label.split('（')[0]}={v[0][i]}" for label, v in results.items())
        print(f"  {fact['topic']:<14} {marks}")


if __name__ == "__main__":
    asyncio.run(main())
