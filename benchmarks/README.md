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


## 端到端问答，以及抽取层（2026-08）

召回率只对存原文的系统公平：一个记忆系统若靠合成作答，它给出的正确答案可能不
对应任何单一原文轮，用「证据轮是否进 top-k」去打分等于用尺子量重量。所以加了
`judge.py`——同一个问题给每个系统，各自返回什么就用什么作答，再由一个不知道答
案出自哪个系统的模型对照标准答案判分。

必须先修的一处：LoCoMo **23% 的答案是日期**，而日期在 `session_N_date_time`
字段里、不在任何轮次文本里。不把会话时间戳带进语料，检索对了也答不出来——补上
之后所有系统的准确率翻倍。

### 结果（2 段对话，120 题，重复 6 次）

| 方案 | 均值 | 极差 |
|---|---|---|
| Cognee（图谱自答） | 64.4% | 2.5 |
| OpenKimo — 换 k3 作答（对照） | 54.4% | 3.3 |
| mem0（qdrant 混合） | 53.5% | 6.6 |
| OpenKimo（检索原文轮次） | 49.2% | 4.1 |
| LlamaIndex BM25 | 48.5% | 9.2 |
| txtai（稠密+BM25） | 48.3% | — |
| Cognee（取回原文块） | 43.9% | 3.3 |

**重复测量的极差最高 9.2 个点，所以小于这个幅度的差距不能当作差异。** 检索是
确定性的，波动全部来自作答与判分的 LLM。据此只有两条结论站得住：

1. Cognee 的图谱路径显著领先。同一个系统、同一份图，取回原文块 43.9%（垫底）
   对图谱自答 64.4%（第一）——二十个点全部来自「用抽取出的结构作答」而非「返回
   原文」，与检索质量无关（它的检索是所有系统里最差的）。
2. 中间四家（我们、mem0、txtai、BM25）互相之间没有可断言的差距。

对照组说明模型本身值约 5 个点（同一份检索结果，换 k3 作答从 49.2% 升到
54.4%），所以 Cognee 的真实领先约十个点而非十五个。

### 抽取层（`extraction_layer.py`）

上面那张表把 OpenKimo 当成纯检索器测，而产品里存的从来不是原文轮次，是抽取并经
用户批准的事实。补上这一层的测量：

| 变体 | 抽出事实 | 准确率 |
|---|---|---|
| 线上提示词，上限 5 | 115 | 6.7% |
| 线上提示词，上限 12 | 141 | 7.5% |
| 改写提示词，上限 12 | 454 | 44.2% |

线上提示词抽的是永恒特质（「Caroline 是跨性别女性」），**一个日期都没有**——
它明写着排除事件（"Exclude anything tied to this conversation: what was done"）。
LoCoMo 问的全是「什么时候发生了什么」，所以 6.7% 说明的是瞄准的目标不同，不是
管线坏了。

有价值的是第三行：换成面向事件的提示词后是 44.2%，而**直接检索原文轮次是
49.2%**——在噪音范围内持平。也就是说「把对话压成扁平事实句」这件事本身不带来
增益。Cognee 领先靠的是我们没有的两样东西：实体与关系的**结构**（519 节点 /
1262 边，我们是 454 条互不相连的句子），以及**时间锚定**（条目内容里带事件发生
时间，而我们只有记录时间）。


## 完整对比（2026-08-19 更新）

同一份 LoCoMo（2 段对话、788 轮、120 题），同一个判分器，所有系统喂完全相同的
带时间戳原文。

| 方案 | 端到端准确率 | 说明 |
|---|---|---|
| Cognee（图谱自答） | **64.4%**（6 次均值） | 用抽取出的实体关系作答 |
| *OpenKimo — 换 k3 作答（对照）* | *53.3%* | 同一份检索结果，只换作答模型 |
| MemOS（取回内容） | 50.8% | LLM 抽取入库 + 纯稠密向量检索 |
| mem0（qdrant 混合） | 49.2% | 向量 + BM25 |
| txtai（稠密+BM25 混合） | 48.3% | |
| LlamaIndex BM25 | 48.3% | |
| OpenKimo（FTS5 + IDF scan） | 46.7% | |
| Cognee（取回原文块） | 45.0% | 同一个 Cognee，不走图谱 |

**重复测量的极差最高 9.2 个点**，所以只有两条结论站得住：

1. **Cognee 的图谱路径是唯一显著领先的**。扣掉作答模型带来的约 5 点，真实领先
   约十点。而它的检索是全场最差（45.0%）——那二十点全部来自「用抽取出的结构
   作答」而非「返回原文」。
2. **中间五家分不出高下，而它们用的技术互不相同**：

   | | 检索方式 |
   |---|---|
   | mem0 52.9% | 稠密向量 + BM25 混合 |
   | MemOS 50.8% | LLM 抽取入库 + 纯稠密向量 |
   | OpenKimo 48.8% | 纯词法（FTS5 三元组 + IDF 子串扫描） |
   | LlamaIndex 48.4% | 纯词法（BM25） |
   | txtai 48.3% | 稠密向量 + BM25 混合 |

   纯词法、纯向量、混合、以及带 LLM 抽取的，全部落在 48–53% 这个五点宽的带里，
   而同一系统重复测量的极差是 9.2 点。**换哪种检索技术都到不了这个带以外** ——
   天花板属于「在对话轮次上做检索」这个范式，不属于某种具体实现。

   我们自己的证据与之一致：今天把中文召回从 47.7% 提到 58.9%（+11 点），端到端
   一点没动。

   （更正：此表早先一版把 MemOS 写成「与我们同技术路线（FTS5 + 向量）」。那来自
   调研阶段一条未经验证的笔记，描述的是它的另一种配置；实际跑的 `general_text`
   后端源码是「嵌入查询、搜向量库」，无词法成分，而我们没有向量。结论不受影响，
   反而更强——共同点不是技术栈，是范式。）

### 没有跑成的，以及原因

| | 阻碍 |
|---|---|
| Graphiti | 嵌入式 Kuzu 后端建图大面积失败（10 轮只出 2 节点 0 边），需 Neo4j + Java |
| Zep | 瘦客户端，需服务端或云 API key |
| Letta | 机制是「对话中 agent 自改写 core memory」，批量灌入测不到 |
| Memori | 机制是拦截 LLM 调用捕获记忆，无公开的批量灌入接口 |
| SimpleMem | 装得上也跑得通，但逐轮串行调 LLM：**64 分钟只答完 20/120 题**，全量需数小时 |
| TencentDB | 未尝试。需起三个 Node 服务，本机 Node 22 具备 |

后四行不是能力评价，是**这套 benchmark 与它们的机制不匹配**，或者代价与信息量
不成比例。用批量灌入去测「对话中维护记忆」的系统，和早先用轮次召回去测图谱是同
一类错误。

### 三处必须连着看的限定

**配置选择两次改变了结论，幅度大于方案之间的差距。** mem0 用其文档推荐的 chroma
时会自行关闭混合检索（英文 68.5% → 58.3%）；中文首次用英文专用嵌入模型加默认
0.1 相似度阈值，它只有 11.6%，据此曾得出「我们中文领先 37 点」，修正后是落后。

**这套 benchmark 测不到本项目的差异化**：零依赖（2 个 JSONL + 1 个可删缓存，
对面 mem0 装出 353MB、txtai 环境 971MB）、写入需用户批准、行为类记忆无需查询即
注入。三条一条都没进表。

**端点怪癖消耗的时间超过任何一次调参。** k3 只接受 `temperature=1`（撞了三次：
对照组全零、cognee 的 tool_choice、SimpleMem 全部写入失败），DeepSeek 不支持
`json_schema`，LiteLLM 要求模型名带 provider 前缀。每一次的表现都是「跑完了，
结果是空的」。


## 这里的检索代码值不值它的体积（`bm25_with_cjk.py`）

LlamaIndex 的 BM25 在英文上与本项目打平，代码量约为三分之一，所以合理的疑问是：
六百多行买到了什么？中文上它 2.3%、我们 58.9%，但这不是答案——**没人需要写六百
行来修这件事**，一个把中文切成双字的分词器约五行。

先说一处失败的测法：`BM25Retriever.from_defaults(tokenizer=...)` 会打印
「deprecated」然后**静默忽略该参数**，结果与默认分词一位不差。若不是数字完全相
同引起怀疑，就会得出「换中文分词也救不回来」——一个恰好对我们有利的错误结论。
改为在喂入前把中文重写为空格分隔的双字，让 LlamaIndex 自带的词干化 BM25 正常
工作。

| 中文 | recall@1 | recall@3 | recall@8 |
|---|---|---|---|
| OpenKimo | **32.8%** | **46.4%** | **58.9%** |
| LlamaIndex BM25（原样） | 0.7% | 2.3% | 2.3% |
| LlamaIndex BM25 + 中文双字预切 | 22.2% | 37.7% | 51.3% |

| 英文 | recall@1 | recall@3 | recall@8 |
|---|---|---|---|
| OpenKimo | **36.4%** | **52.6%** | **64.6%** |
| LlamaIndex BM25 | 30.8% | 47.4% | 60.6% |

结算下来：

- **中文能力的主体不是这些代码买的。** 2.3% → 51.3%（49 个点）来自一个五行的
  分词决定；本项目额外的机制（IDF 加权、语言分流、倒排表）再加 7.6 点。
- **但那 7.6 点是真的。** 检索召回是确定性测量，不受端到端指标那 9.2 点的 LLM
  波动影响；两个独立的 BM25 实现（手写的与 LlamaIndex 的）在中文上都落在
  51.3%，互相印证。
- **它换不成答案质量。** 端到端上本项目 48.8%、LlamaIndex BM25 48.4%，分不出
  高下——与「中文召回 +11 点、端到端纹丝不动」是同一个现象。

所以诚实的定位是：检索确实比最好的 BM25 强，强得可测但不足以改变答案；这些代码
真正值钱的地方是零依赖（仅标准库 `sqlite3`）与增量索引（两万条时写入后再搜索
50 ms，全量重建则是 690 ms），而这两条没有任何一个 benchmark 在测。
