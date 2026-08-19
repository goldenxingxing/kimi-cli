"""Judge the same fixed answers repeatedly, to separate two sources of noise.

The end-to-end table's spread was up to ten points, and it was never clear how
much of that was the systems and how much the judge. For a system that answers
for itself the answers are fixed, so re-judging the same file isolates the
judge exactly — and SimpleMem, the largest number in the table and the one a
design decision would rest on, had been run once.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from judge import _JUDGE_PROMPT, Model

ROUNDS = int(os.environ.get("REJUDGE_ROUNDS", "5"))


async def one_round(model: Model, client: httpx.AsyncClient, rows) -> float:
    async def mark(row) -> float:
        given = (row.get("graph_answer") or row.get("notes") or "").strip()
        if not given:
            return 0.0
        verdict = await model.ask(
            client,
            _JUDGE_PROMPT.format(question=row["question"], gold=row["gold"], given=given),
        )
        return 1.0 if re.match(r"\s*CORRECT", verdict or "", re.I) else 0.0

    return statistics.mean(await asyncio.gather(*(mark(r) for r in rows))) * 100


async def main() -> None:
    files = [(label, Path(path)) for label, path in (s.split("=", 1) for s in sys.argv[1:])]
    model = Model(8)
    async with httpx.AsyncClient(timeout=180) as client:
        print(f"每个方案判 {ROUNDS} 轮，答案固定不变\n")
        print(f"{'方案':<26}{'均值':>9}{'标准差':>9}{'极差':>8}   各轮")
        print("-" * 76)
        for label, path in files:
            rows = json.loads(path.read_text(encoding="utf-8"))
            scores = [await one_round(model, client, rows) for _ in range(ROUNDS)]
            spread = max(scores) - min(scores)
            each = " ".join(f"{s:.1f}" for s in scores)
            sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"{label:<24}{statistics.mean(scores):>8.1f}%{sd:>8.2f}{spread:>8.1f}   {each}")


if __name__ == "__main__":
    asyncio.run(main())
