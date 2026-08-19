"""How often the agent actually reads its memory, from real sessions.

Every other measurement here asks whether retrieval *would* help. This one asks
whether it is reached at all, which turned out to be the binding constraint:
across seven real sessions the Memory tool was called once, and that once was
an `add`. A retriever nobody calls cannot be improved by making it better.

Run before and after a change to the tool description. It reads the session
files the product already writes — no instrumentation, and nothing to enable.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

SESSIONS = Path(
    os.environ.get("SESSION_DATA", Path.home() / "Documents/local_agent_work/session-data")
)

#: Reading is what this measures; writing was never the problem.
_READS = {"search", "get", "list"}


def tool_calls(path: Path):
    """`(tool name, memory op or None)` for every call recorded in a session."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("role") != "assistant":
            continue
        for call in row.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            name = str(fn.get("name") or call.get("name") or "?")
            op = None
            if name.lower() == "memory":
                args = fn.get("arguments") or call.get("input") or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                operation = args.get("operation") if isinstance(args, dict) else None
                op = str(operation.get("op") if isinstance(operation, dict) else operation)
            yield name, op


def main() -> None:
    if not SESSIONS.is_dir():
        print(f"没有找到会话目录：{SESSIONS}")
        sys.exit(1)

    tools: Counter[str] = Counter()
    ops: Counter[str] = Counter()
    sessions = 0
    for directory in sorted(SESSIONS.iterdir()):
        context = directory / "context.jsonl"
        if not context.is_file():
            continue
        sessions += 1
        for name, op in tool_calls(context):
            tools[name] += 1
            if op is not None:
                ops[op] += 1

    total = sum(tools.values())
    reads = sum(n for op, n in ops.items() if op in _READS)
    print(f"{sessions} 个会话，{total} 次工具调用\n")
    print(f"{'工具':<24}{'次数':>7}{'占比':>9}")
    print("-" * 42)
    for name, n in tools.most_common(10):
        print(f"{name:<24}{n:>7}{n / max(1, total) * 100:>8.1f}%")

    print(f"\nMemory 操作：{dict(ops) or '无'}")
    print(
        f"读取类（search/get/list）：{reads} 次"
        f" —— 每 {sessions} 个会话 {reads / max(1, sessions):.2f} 次"
    )
    if not reads:
        print("\n一次都没有。改动是否有效，看这个数字有没有离开零。")


if __name__ == "__main__":
    main()
