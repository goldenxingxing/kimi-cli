"""How many real questions here need more than one remembered fact?

Cognee's graph is ahead by about ten points once the answering model is
controlled for, and the graph earns that on LoCoMo — dense personal narrative
where answers are assembled from several turns. Whether it transfers depends on
something no benchmark can say: the shape of the questions actually asked here.

Assuming it transfers is the mistake already made twice today, so this counts
instead. Real user turns are classified into three kinds, because the middle
one is what a graph would help with and the other two are not:

- an instruction, which memory is not being asked to answer at all
- answerable from one recorded fact
- needing two or more combined

A graph pays for itself only if the third kind is common. If it is rare, the
ten points are for a question shape that does not occur here.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

import httpx

#: Reads local transcripts to count what the user actually asks for. Nothing
#: leaves the machine except the turns themselves, one per classification call.
#:
#: OpenKimo's own session files are post-compaction tails and hold thirteen real
#: user turns between them — not enough to count anything. The Claude Code
#: transcripts for the same directories are the same body of work, uncompacted.
PROJECTS = Path.home() / ".claude" / "projects"
SAMPLE = int(os.environ.get("HOP_SAMPLE", "300"))

_PROMPT = """\
下面是一位用户对编程助手说的一句话。判断回答它需要用到什么，只回一个字母。

关键区分：**跨会话记忆**指的是几周几月前另一次对话里定下的事实，助手现在看不到
原始对话，只能靠存下来的记录。当前对话里刚说过的、代码里读得到的、报错信息里写
着的，都**不算**。

A = 不需要跨会话记忆
    · 指令或任务请求（"改一下"、"跑测试"、"提交"、"发版"）
    · 在问当前这次对话里已经发生的事（"你刚才说的那个结果呢"）
    · 在报 bug 或描述现象，助手去看代码/日志就能答
    · 在问代码逻辑，助手读一遍源码就能答

B = 需要跨会话记忆，且**一条**存下来的事实就够
    （某个决定是什么、某个路径在哪、某条约定是什么）

C = 需要跨会话记忆，且必须把**两条或更多分别存下来**的事实组合起来
    （比较两个分支/方案各自的记录、追溯某个决定与另一个决定的关系、
      「当时为什么这么定」需要串起前因与后果）

拿不准时选 A。只回 A、B 或 C。

用户的话：{turn}"""


def user_turns() -> list[str]:
    """Everything the user actually typed, across every recorded project."""
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
                # Tool results and pasted output are not things the user asked.
                if len(text) < 4 or len(text) > 300 or text.count("\n") > 6:
                    continue
                out.append(text)
    return out


async def classify(client: httpx.AsyncClient, sem: asyncio.Semaphore, turn: str) -> str:
    async with sem:
        for _ in range(3):
            try:
                r = await client.post(
                    os.environ["BENCH_BASE_URL"].rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                    json={
                        "model": os.environ["BENCH_MODEL"],
                        "messages": [{"role": "user", "content": _PROMPT.format(turn=turn)}],
                        "temperature": 0,
                    },
                )
                r.raise_for_status()
                reply = r.json()["choices"][0]["message"]["content"].strip().upper()
                match = re.search(r"\b([ABC])\b", reply)
                if match:
                    return match.group(1)
            except Exception:
                await asyncio.sleep(1.0)
    return "?"


async def main() -> None:
    everything = user_turns()
    # Evenly spaced rather than the first N: recent work would otherwise be the
    # only thing measured, and the question is about how this user works.
    step = max(1, len(everything) // SAMPLE)
    turns = everything[::step][:SAMPLE]
    print(f"记录中共 {len(everything)} 条用户发言，等距取样 {len(turns)} 条\n")

    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=120) as client:
        labels = await asyncio.gather(*(classify(client, sem, t) for t in turns))

    counts = Counter(labels)
    asked = counts["B"] + counts["C"]
    print(f"{'类别':<44}{'条数':>6}{'占比':>9}")
    print("-" * 62)
    for key, name in (
        ("A", "指令/任务请求（记忆不参与）"),
        ("B", "问事实，单条记录可答"),
        ("C", "问事实，需组合两条以上"),
        ("?", "分类失败"),
    ):
        n = counts[key]
        print(f"{name:<40}{n:>6}{n / len(turns) * 100:>8.1f}%")

    if asked:
        print(f"\n在「确实在问事实」的 {asked} 条里，需要多跳的占 {counts['C'] / asked * 100:.1f}%")
    # Dumped in full, because the count is only worth as much as the labels
    # and one visibly wrong label has already turned up: a bare list of branch
    # names was filed as a multi-hop question.
    out = Path(os.environ.get("HOP_DUMP", "/tmp/hop_labels.txt"))
    with out.open("w", encoding="utf-8") as handle:
        for turn, label in zip(turns, labels, strict=True):
            handle.write(f"{label}\t{turn}\n")
    print(f"\n全部标注写入 {out}")

    print("\n判为多跳的前 15 条：")
    for turn, label in list(zip(turns, labels, strict=True)):
        if label == "C":
            print(f"  - {turn[:92]}")


if __name__ == "__main__":
    asyncio.run(main())
