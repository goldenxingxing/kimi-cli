"""If every turn triggered a retrieval, how often would it help — and how often
would it just add noise?

The proposal is to stop injecting a fixed index at session start and instead
retrieve per turn. Before changing that, there is no baseline: the `search`
operation is called essentially never in real sessions, so "did it improve" has
nothing to improve on.

Three things have to be separated, because they call for different fixes:

- **有没有** — does the store even contain something that would help this turn?
  This is the ceiling. Retrieval cannot beat it.
- **捞不捞得到** — when something does exist, does our search surface it?
  This is a retrieval problem and is the only one more retrieval work can fix.
- **噪音** — when nothing relevant exists, does search hand back entries anyway?
  Auto-injection pays this on every turn that had nothing to find, which — if
  the instruction share holds — is most of them.

The store is built from the user's own sessions with the shipped extraction
prompt, so it is the store this design would actually be running against.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.search import MemorySearchIndex

PROJECTS = Path.home() / ".claude" / "projects"
SESSIONS = Path("/Users/qunwei/Documents/local_agent_work/session-data")
TURNS = int(os.environ.get("TURN_SAMPLE", "120"))
TOP_K = 3


def user_turns() -> list[str]:
    out: list[str] = []
    for path in sorted(PROJECTS.glob("*/*.jsonl")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in raw.splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            message = row.get("message") or {}
            if row.get("type") != "user" or message.get("role") != "user":
                continue
            content = message.get("content")
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in parts:
                if not isinstance(part, dict) or part.get("type") not in (None, "text"):
                    continue
                text = (part.get("text") or "").strip()
                if not text or "system-reminder" in text or text.startswith(("<", "[")):
                    continue
                if len(text) < 6 or len(text) > 300 or text.count("\n") > 6:
                    continue
                out.append(text)
    return out


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


_EXISTS = """\
下面是一个人的长期记忆库（跨会话记下的事实），以及他刚说的一句话。

判断：库里**有没有**哪一条，如果放进上下文，会实质帮助理解或执行这句话？
「实质帮助」指的是提供了这句话本身没有、而助手又需要的信息（某个路径、某个决定、
某条约定）。仅仅主题沾边不算。

只回一个数字：帮得上的条目编号（多个用逗号分隔），没有就回 0。

记忆库：
{store}

他说的话：{turn}"""

_USEFUL = """\
下面是一个人刚说的一句话，以及检索系统为这句话取回的几条长期记忆。

对每一条判断它是否会实质帮助理解或执行这句话——提供了这句话本身没有、而助手又
需要的信息。仅仅主题沾边不算，宁可判否。

按顺序只回同样数量的 Y 或 N，逗号分隔，比如 `N,Y,N`。

他说的话：{turn}

取回的记忆：
{hits}"""


async def build_store(client, sem) -> list[str]:
    """Extract facts from the user's own sessions with the shipped prompt."""
    import time as _time

    from try_extract import conversation_tail

    from kimi_cli.memory.archivist import _EXTRACTION_PROMPT, _parse_candidates

    ids = sorted(d.name for d in SESSIONS.iterdir() if (d / "context.jsonl").is_file())
    today = _time.strftime("%Y-%m-%d", _time.localtime())
    replies = await asyncio.gather(
        *(
            ask(
                client,
                sem,
                _EXTRACTION_PROMPT.format(conversation=conversation_tail(i), today=today),
            )
            for i in ids
        )
    )
    facts: list[str] = []
    for reply, sid in zip(replies, ids, strict=True):
        facts += [c.content for c in _parse_candidates(reply, session_id=sid)]
    return facts


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items))


async def main() -> None:
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=180) as client:
        facts = await build_store(client, sem)
        print(f"记忆库：从 {len(list(SESSIONS.iterdir()))} 个真实会话抽出 {len(facts)} 条\n")
        if not facts:
            return

        entries = [MemoryEntry(kind="project", scope="persistent", content=f) for f in facts]
        text = {e.id: e.content for e in entries}
        tmp = Path(tempfile.mkdtemp())
        (tmp / "src").write_text("x", encoding="utf-8")
        index = MemorySearchIndex(tmp / "db", tmp / "src")

        everything = user_turns()
        step = max(1, len(everything) // TURNS)
        turns = everything[::step][:TURNS]
        store_text = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))

        exists = await asyncio.gather(
            *(ask(client, sem, _EXISTS.format(store=store_text, turn=t)) for t in turns)
        )

        retrieved = [
            [text[h.entry_id] for h in index.search(t, entries, limit=TOP_K)] for t in turns
        ]
        useful = await asyncio.gather(
            *(
                ask(client, sem, _USEFUL.format(turn=t, hits=numbered(hits)))
                if hits
                else asyncio.sleep(0, result="")
                for t, hits in zip(turns, retrieved, strict=True)
            )
        )

    counts = Counter()
    hit_when_exists = 0
    noise_rows = 0
    injected = 0
    injected_useful = 0
    for reply_exists, hits, reply_useful in zip(exists, retrieved, useful, strict=True):
        wanted = {int(x) for x in re.findall(r"\d+", reply_exists or "0")} - {0}
        has = bool(wanted)
        counts["库里有相关的" if has else "库里没有相关的"] += 1

        marks = [m.upper() for m in re.findall(r"[YN]", reply_useful or "")]
        good = sum(1 for m in marks[: len(hits)] if m == "Y")
        injected += len(hits)
        injected_useful += good
        if has and good:
            hit_when_exists += 1
        if not has and hits:
            noise_rows += 1

    total = len(turns)
    has_n = counts["库里有相关的"]
    print(f"{total} 条真实用户发言，每轮取 top-{TOP_K}\n")
    print(f"  ① 库里存在相关记忆          : {has_n:>4} / {total}  ({has_n / total * 100:.1f}%)")
    if has_n:
        rate = hit_when_exists / has_n * 100
        print(f"  ② 其中检索捞到了            : {hit_when_exists:>4} / {has_n}  ({rate:.1f}%)")
    print(f"  ③ 库里没有、检索仍返回内容  : {noise_rows:>4} / {total - has_n}")
    print()
    print(
        f"  自动注入的话：共 {injected} 条，其中判定有用 {injected_useful} 条 "
        f"（{injected_useful / max(1, injected) * 100:.1f}%）"
    )


if __name__ == "__main__":
    asyncio.run(main())
