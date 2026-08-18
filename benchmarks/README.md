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
ranking. The numbers below are English-only — LoCoMo is an English dataset, and
the Chinese path has unit tests but no benchmark.

| | recall@1 | recall@5 | recall@10 |
| --- | --- | --- | --- |
| whole question | 35.3% | 57.0% | 65.3% |
| BM25 baseline | 26.1% | 49.3% | 57.9% |

[locomo]: https://github.com/snap-research/locomo
