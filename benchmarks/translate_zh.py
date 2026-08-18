"""Build a Chinese LoCoMo by translation, keeping the evidence labels.

The claim that our CJK handling beats a BM25 tokenizer is, so far, a guess.
Testing it needs a Chinese set with the same gold labels — and translation is
the one way to get that while holding everything else fixed: the conversations,
the questions, and the dia_id evidence are unchanged, so any difference in the
scores is language and nothing else.

Scoring never looks at answer strings (it asks whether the gold *turn* came
back), so translation cannot corrupt the ground truth the way it would in a
string-matching benchmark.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
SRC = HERE / "locomo10.json"
OUT = HERE / "locomo10_zh.json"
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "2"))
BATCH = 25
CONCURRENCY = 6

_PROMPT = """\
把下面编号的英文句子逐条译成自然的简体中文。人名保留英文原样。

严格按 JSON 对象返回，键是编号字符串，值是译文，不要有任何其他内容：
{{"1": "...", "2": "..."}}

{lines}"""


async def translate(
    client: httpx.AsyncClient, texts: list[str], sem: asyncio.Semaphore
) -> list[str]:
    if not texts:
        return []
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(
                    os.environ["BENCH_BASE_URL"].rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                    json={
                        "model": os.environ["BENCH_MODEL"],
                        "messages": [{"role": "user", "content": _PROMPT.format(lines=lines)}],
                        "response_format": {"type": "json_object"},
                    },
                )
                r.raise_for_status()
                got = json.loads(r.json()["choices"][0]["message"]["content"])
                out = [str(got.get(str(i + 1), texts[i])) for i in range(len(texts))]
                if all(out):
                    return out
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"    批次失败，保留英文原文: {exc}", file=sys.stderr)
    return texts  # untranslated is better than lost; counted and reported


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def main() -> None:
    samples = json.loads(SRC.read_text(encoding="utf-8"))[:SAMPLES]
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=240) as client:
        for n, sample in enumerate(samples):
            convo = sample["conversation"]
            names = [k for k in convo if k.startswith("session_") and "date" not in k]

            turns = [t for name in names for t in convo[name] if t.get("text")]
            texts = [t["text"] for t in turns]
            questions = [qa for qa in sample.get("qa", []) if (qa.get("question") or "").strip()]

            jobs = [translate(client, c, sem) for c in _chunks(texts, BATCH)]
            jobs += [
                translate(client, [q["question"] for q in c], sem)
                for c in _chunks(questions, BATCH)
            ]
            done = await asyncio.gather(*jobs)

            flat = [x for part in done for x in part]
            for turn, zh in zip(turns, flat[: len(texts)], strict=True):
                turn["text"] = zh
            for qa, zh in zip(questions, flat[len(texts) :], strict=True):
                qa["question"] = zh

            untouched = sum(1 for a, b in zip(texts, flat[: len(texts)], strict=True) if a == b)
            print(
                f"  样本 {n + 1}/{len(samples)}: {len(texts)} 轮 + {len(questions)} 问"
                f"（{untouched} 轮未译）",
                flush=True,
            )

    OUT.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
