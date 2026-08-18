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


## vs_mem0_retrieval.py — 与 mem0 的检索对比

只比检索器，不比整个记忆系统。两边存入完全相同的原文（mem0 用 `infer=False`
关掉它的 LLM 抽取），打分用 LoCoMo 自带的 `evidence` 证据轮标注，不看答案字符串
——否则"存原文"的一方会因为标准答案本就是原文子串而白捡分数。

这也意味着它**测不到 mem0 的抽取层**，而那是 mem0 的核心。任何"谁的记忆系统更好"
的结论都不能从这个脚本得出。

运行（mem0 不是本项目依赖）：

    BENCH_SYS_PATH=/path/to/mem0-install python benchmarks/vs_mem0_retrieval.py

中文对比需要先用翻译脚本生成 `locomo10_zh.json`：证据标注是轮次 id，翻译不会
破坏它，所以除语言外所有变量都被控制住。中文必须配多语言嵌入模型
（`BENCH_EMBED`），用英文专用模型测中文测的是模型选型不是检索能力。

### 两个必须设对的参数

mem0 的 `search()` 参数是 `top_k` 不是 `limit`；传 `limit=` 会被 `**kwargs`
静默吞掉。`threshold` 默认 0.1，中文场景下正确命中的分数常在 0.101 附近，默认值
会把结果砍到只剩一条——那时测的是阈值，不是召回。两处都设错过，都产生过看起来
可信但完全错误的结论。

### 结果（2 段对话，788 轮，302 问，2026-08）

英文：

| 检索器 | recall@1 | recall@3 | recall@8 |
|---|---|---|---|
| OpenKimo (FTS5 词法) | 36.4% | 52.6% | 64.6% |
| mem0 纯向量 (bge-small-en) | 25.5% | 44.4% | 58.3% |
| mem0 混合 (向量+BM25) | 36.4% | 57.3% | **68.5%** |

中文（同批对话译入，证据标注不变）：

| 检索器 | recall@1 | recall@3 | recall@8 |
|---|---|---|---|
| OpenKimo (FTS5 词法) | 19.5% | 34.8% | 47.7% |
| mem0 纯向量 (多语言 MiniLM) | 20.2% | 38.4% | **54.6%** |
| mem0 混合 (向量+BM25) | 18.5% | 31.8% | 49.3% |

结论：**纯词法检索在两种语言上都落后于带稠密向量的方案**（英文 -3.9、中文
-6.9，均为 recall@8，也是 `search` 工具实际返回的条数）。BM25 那一半在英文上
值 +10 个点，在中文上是负收益（49.3 对 54.6）——空格分词在中文上不成立，这一点
与选哪个嵌入模型无关。

我们自己的中文比英文低 17 个点（47.7 对 64.6）：中文检索是这套方案最弱的一环。

### 中文排序修复后（2026-08）

scan 此前按「命中了几个 needle」排序，而中文 needle 是查询的每个双字窗口——
问句的语法（什么、时候、怎么）与内容（互助、小组）等权。英文靠停用词表挡住了
这件事，中文没有表。改为按逆文档频率加权，并让含中文的查询以 scan 为主、FTS5
只补空位（纯拉丁查询走原路径不变）：

| | recall@1 | recall@3 | recall@8 |
|---|---|---|---|
| 中文 修改前 | 19.5% | 34.8% | 47.7% |
| 中文 修改后 | 32.8% | 46.4% | **58.9%** |
| 英文 修改前后 | 36.4% | 52.6% | 64.6% |

中文 +11.2 点（recall@8），英文一点不动，仍然零依赖。对比 mem0：中文 58.9%
对 54.6%（领先），英文 64.6% 对 68.5%（落后）。

调优过程中有一个必须记下来的教训：最初的版本还会剪掉出现在 5% 以上条目里的
needle，在两千条的 benchmark 上多赚半个点，在真实记忆库上是灾难——真实库只有
几十条，5% 不足 1，于是**凡是真正匹配上的 needle 都被剪掉**，只留下匹配不到
任何东西的。benchmark 抓不到这个，因为它从没在真实规模的库上跑过。
`TestRankingOnASmallStore` 就是为这件事留的。
