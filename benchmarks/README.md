# Memory retrieval benchmark

Answers one question: when the agent goes looking for something it stored, does
it find it? Nothing here calls a model — [LoCoMo][locomo] labels which dialogue
turn holds each answer, so a hit is a fact rather than a judgement, and the
whole thing runs offline in about a minute.

```bash
curl -sSLo benchmarks/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

python benchmarks/memory_recall.py    # our search
python benchmarks/bm25_baseline.py    # BM25, for reference
```

Every dialogue turn becomes one memory entry; every question becomes one query.
A question counts as answered if the labelled turn comes back at all.

## Why the baseline is here

A recall figure on its own cannot say whether the number is the task being hard
or the engine being weak, and those call for different responses. BM25 is the
standard keyword baseline and needs no embeddings or services — the same class
of thing this project could ship — so it marks what is reachable without
changing the architecture.

## What it has caught

The first run scored **0.0% at every depth**. The whole query was being matched
as a single quoted phrase, which meant a natural question could only match a
turn containing that question verbatim. Quoting had been added to stop FTS5
reading `core.py` as syntax; it also made the engine useless, and nothing said
so until this was run. Splitting the query into OR-ed terms took recall@10 to
**65.3%**, past BM25's 57.9%.

Worth re-running after any change to query construction, tokenisation, or
ranking.

| | recall@1 | recall@5 | recall@10 |
| --- | --- | --- | --- |
| whole question | 35.3% | 57.0% | 65.3% |
| BM25 baseline | 26.1% | 49.3% | 57.9% |

## Bilingual

```bash
python benchmarks/bilingual_recall.py
```

LoCoMo is English, and English alone is not what this has to work for: a real
store mixes Chinese prose with English identifiers inside single entries and is
queried the same way. That path arrived as a fallback — used only when the index
came back empty — so a mixed query answered its English half and dropped the
Chinese one, which is the commonest shape of query there is.

The fixture is written by hand; there is no public Chinese long-conversation
memory set to sample from. So this is a regression target rather than a
comparison: it says whether a change made retrieval worse, not whether
retrieval is good.

Several queries expect **two** entries — one matched by an English identifier,
one by Chinese prose — and score only if both come back. That detail is the
whole point. The first version scored each query on finding any one relevant
entry, and gave an identical 100% before and after the bilingual fix; it was
worthless until it was rewritten to fail.

[locomo]: https://github.com/snap-research/locomo
