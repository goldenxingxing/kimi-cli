"""BM25 over the same data, as a reference point.

Without it the recall figure means nothing: it could be the task being hard or
the engine being weak, and those call for different responses. BM25 is the
standard keyword baseline — no embeddings, no service, the same class of thing
we could plausibly ship.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

DATA = Path(__file__).with_name("locomo10.json")
"""Download: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"""
TOP_K = (1, 3, 8)  # match memory_recall.py — the depth the tool returns
K1, B = 1.5, 0.75


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


class BM25:
    def __init__(self, docs: list[list[str]]) -> None:
        self.docs = docs
        self.freqs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / max(len(docs), 1)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def top(self, query: list[str], k: int) -> list[int]:
        scores: list[tuple[float, int]] = []
        for i, freq in enumerate(self.freqs):
            s = 0.0
            for t in query:
                f = freq.get(t, 0)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self.lens[i] / self.avg)
                s += self.idf.get(t, 0.0) * f * (K1 + 1) / denom
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


def main() -> None:
    with DATA.open(encoding="utf-8") as handle:
        samples = json.load(handle)
    per_k = {k: [] for k in TOP_K}
    asked = 0

    for sample in samples:
        convo = sample["conversation"]
        docs, ids = [], []
        for name in sorted(k for k in convo if k.startswith("session_") and "date" not in k):
            for turn in convo[name]:
                if turn.get("text"):
                    docs.append(tokens(f"{turn['speaker']}: {turn['text']}"))
                    ids.append(turn["dia_id"])
        engine = BM25(docs)

        for qa in sample.get("qa", []):
            question = (qa.get("question") or "").strip()
            gold = {e for e in (qa.get("evidence") or []) if isinstance(e, str)}
            if not question or not gold:
                continue
            asked += 1
            ranked = engine.top(tokens(question), max(TOP_K))
            got = [ids[i] for i in ranked]
            for k in TOP_K:
                per_k[k].append(1.0 if gold & set(got[:k]) else 0.0)

    print(f"BM25 基线（同样 {asked} 个问题，整句查询）\n")
    print(f"{'指标':<14}{'召回率':>10}")
    print("-" * 26)
    for k in TOP_K:
        print(f"{'recall@' + str(k):<14}{statistics.mean(per_k[k]) * 100:>9.1f}%")


if __name__ == "__main__":
    main()
