"""Retrieval over bilingual memory, which is the shape a real store takes.

LoCoMo is English, and English alone is not the case this has to work for: an
actual store mixes Chinese prose with English identifiers — file paths, branch
names, tool names — inside single entries, and asks questions the same way.
Chinese arrived here as a fallback path rather than a first-class one, and a
fallback nobody measures is a fallback nobody notices breaking.

The fixture is written rather than sampled, because there is no public Chinese
long-conversation memory set to sample from. That makes this a regression test
against a fixed target, not a benchmark against other systems — it says whether
a change made retrieval worse, not whether retrieval is good.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kimi_cli.memory.entry import MemoryEntry
from kimi_cli.memory.search import MemorySearchIndex

TOP_K = (1, 3, 5)

#: (key, body). Deliberately in the register real entries are written in:
#: Chinese sentences carrying English identifiers, several entries per topic so
#: that matching the topic is not enough to score.
CORPUS: list[tuple[str, str]] = [
    ("acls/repo", "acls 项目真实仓库路径为 ~/Documents/acls，不是 output/ 下面的副本"),
    ("acls/branch", "acls 当前工作分支是 kimo_perf_backport，切换分支后 CodeGraph 必须 sync"),
    ("acls/codegraph", "CodeGraph 索引不感知 git 分支，只反映当前检出的文件快照"),
    ("kimi/config", "OpenKimo 桌面版后端以 KIMI_SHARE_DIR 指向 sessions 目录读取配置"),
    ("kimi/port", "OpenKimo 后端默认监听 5494 端口，被占用时不会自动换端口"),
    ("kimi/skills", "内置 skill 目录在包内 skills/，用户目录是 ~/.claude/skills"),
    ("mail/126", "126 邮箱已接入并验证可用，配置在 skill 目录下的 .env 文件里"),
    ("mail/smtp", "发信走 SMTP 587 端口，必须使用授权码而不是登录密码"),
    ("vault/linker", "output 目录的 Obsidian vault 链接层由 .tools/vault_linker.py 生成"),
    ("vault/watch", "vault_watch 自动监听已上线，写完 md 不需要手动跑 linker"),
    ("pump/closed-loop", "闭环控制算法设计文档在 output/closed_loop_algorithm_design/ 下"),
    ("pump/user", "用户是胰岛素泵厂家，第三方血糖管理 App 想接入他们的泵"),
    ("bench/locomo", "LoCoMo dataset lives in benchmarks/ and is gitignored"),
    ("bench/bm25", "The BM25 baseline exists so a recall number can be interpreted"),
]

#: (query, expected keys). Mixed on purpose: Chinese questions, English
#: questions, and the mixed form that is most common in practice.
#:
#: Several expect *both* an English-identifier match and a Chinese-prose match
#: from one query. Those are the ones that matter: a query is scored on finding
#: every expected entry, so an engine that answers the English half and drops
#: the Chinese one fails, where scoring "did anything relevant come back" would
#: have called it a pass. It did, until this was written that way.
QUERIES: list[tuple[str, tuple[str, ...]]] = [
    # Chinese only.
    ("acls 的仓库路径在哪", ("acls/repo",)),
    ("切换分支之后要做什么", ("acls/branch",)),
    ("端口被占用会怎么样", ("kimi/port",)),
    ("邮箱是怎么配置的", ("mail/126",)),
    ("发邮件用什么密码", ("mail/smtp",)),
    ("写完 markdown 还要手动跑脚本吗", ("vault/watch",)),
    ("闭环算法的设计文档在哪", ("pump/closed-loop",)),
    ("用户是做什么的", ("pump/user",)),
    # English only.
    ("where does the LoCoMo dataset live", ("bench/locomo",)),
    ("why is there a BM25 baseline", ("bench/bm25",)),
    # Mixed, and expecting both halves. An engine that stops at the English
    # identifier answers the first key and misses the second.
    ("CodeGraph 有什么限制", ("acls/codegraph", "acls/branch")),
    ("vault_watch 还要手动生成链接吗", ("vault/watch", "vault/linker")),
    ("KIMI_SHARE_DIR 指向哪个目录", ("kimi/config",)),
    ("SMTP 用哪个端口，要密码吗", ("mail/smtp",)),
    ("acls 分支和 CodeGraph 的关系", ("acls/branch", "acls/codegraph")),
]


def main() -> None:
    entries = [
        MemoryEntry(kind="project", scope="persistent", key=key, content=body)
        for key, body in CORPUS
    ]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "source.jsonl").write_text("fixture", encoding="utf-8")
    index = MemorySearchIndex(tmp / "search.sqlite3", tmp / "source.jsonl")

    per_k = {k: [] for k in TOP_K}
    misses: list[str] = []
    for query, expected in QUERIES:
        hits = [h.handle for h in index.search(query, entries, limit=max(TOP_K))]
        for k in TOP_K:
            # Every expected entry must be there, not just one of them.
            per_k[k].append(1.0 if set(expected) <= set(hits[:k]) else 0.0)
        if not set(expected) <= set(hits[: max(TOP_K)]):
            lost = sorted(set(expected) - set(hits[: max(TOP_K)]))
            misses.append(f"{query}  →  漏掉 {lost}，得到 {hits[:3] or '(无)'}")

    print(f"双语检索：{len(CORPUS)} 条记忆，{len(QUERIES)} 个查询\n")
    print(f"{'指标':<12}{'命中率':>10}")
    print("-" * 24)
    for k in TOP_K:
        print(f"{'recall@' + str(k):<12}{statistics.mean(per_k[k]) * 100:>9.1f}%")
    if misses:
        print(f"\n未命中 {len(misses)} 条：")
        for line in misses:
            print("  ", line)


if __name__ == "__main__":
    main()
