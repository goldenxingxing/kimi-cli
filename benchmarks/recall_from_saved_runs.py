"""Recall for the two systems whose runs were scored end-to-end only.

Both wrote out what they retrieved, so this needs no re-run — only mapping the
returned text back to the turn ids the gold evidence is written in.

They are not equally comparable, and saying so is the point. MemOS returns
exactly eight verbatim turns, which is what every other row in the recall table
returns. Cognee's chunk path returns eight *documents* of twenty turns each:
scoring that as recall@8 would let it answer with a hundred and sixty turns
against everyone else's eight, and would look like a twenty-fold advantage that
is entirely a difference in what a "result" means.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

TOP_K = (1, 3, 8)
HERE = Path(__file__).parent
DATA = HERE / os.environ.get("BENCH_DATA", "locomo10.json")
SAMPLES = 2


def gold_and_turns():
    """`(question -> evidence ids, normalized turn text -> dia_id)`."""
    gold: dict[str, set[str]] = {}
    by_text: dict[str, str] = {}
    for sample in json.loads(DATA.read_text(encoding="utf-8"))[:SAMPLES]:
        convo = sample["conversation"]
        names = sorted(
            (k for k in convo if k.startswith("session_") and "date" not in k),
            key=lambda n: int(n.split("_")[1]),
        )
        for name in names:
            for t in convo[name]:
                if t.get("text") and t.get("dia_id"):
                    by_text[t["text"].strip()[:60]] = t["dia_id"]
        for qa in sample.get("qa", []):
            q = (qa.get("question") or "").strip()
            ev = {str(e) for e in (qa.get("evidence") or [])}
            if q and ev:
                gold[q] = ev
    return gold, by_text


def dia_ids(note_lines, by_text):
    out = []
    for line in note_lines:
        body = line.split(": ", 1)[-1].strip()[:60]
        found = by_text.get(body)
        if found:
            out.append(found)
    return out


def score(path: Path, label: str, *, per_result: int) -> None:
    gold, by_text = gold_and_turns()
    rows = json.loads(path.read_text(encoding="utf-8"))
    scored = []
    for row in rows:
        want = gold.get(row["question"])
        if not want:
            continue
        lines = [x for x in (row.get("notes") or "").split("\n") if x.strip()]
        ranked = dia_ids(lines, by_text)
        scored.append({k: 1.0 if want & set(ranked[: k * per_result]) else 0.0 for k in TOP_K})
    if not scored:
        print(f"{label}: 无可评分的问题")
        return
    cells = "".join(f"{statistics.mean(r[k] for r in scored) * 100:>10.1f}%" for k in TOP_K)
    suffix = "" if per_result == 1 else f"（每个结果 {per_result} 轮，非同一预算）"
    print(f"{label:<34}{cells}   n={len(scored)}{suffix}")


def ours_on(questions) -> list[dict[int, float]]:
    """This project scored on exactly the questions the other two answered.

    Their runs used the first sixty QA pairs per conversation; the recall table
    elsewhere uses all three hundred and two with evidence. Comparing across
    those two sets would be comparing question difficulty, so the baseline is
    recomputed here rather than carried over.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import tempfile

    from kimi_cli.memory.entry import MemoryEntry
    from kimi_cli.memory.search import MemorySearchIndex

    turns = []
    for sample in json.loads(DATA.read_text(encoding="utf-8"))[:SAMPLES]:
        convo = sample["conversation"]
        names = sorted(
            (k for k in convo if k.startswith("session_") and "date" not in k),
            key=lambda n: int(n.split("_")[1]),
        )
        for name in names:
            when = convo.get(f"{name}_date_time", "")
            for t in convo[name]:
                if t.get("text") and t.get("dia_id"):
                    stamp = f"[{when}] " if when else ""
                    turns.append((t["dia_id"], f"{stamp}{t['speaker']}: {t['text']}"))

    entries = [MemoryEntry(kind="project", scope="persistent", content=x) for _, x in turns]
    dia = {e.id: d for e, (d, _) in zip(entries, turns, strict=True)}
    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").write_text("x", encoding="utf-8")
    index = MemorySearchIndex(tmp / "db", tmp / "src")

    out = []
    for q, want in questions:
        ranked = [dia[h.entry_id] for h in index.search(q, entries, limit=max(TOP_K))]
        out.append({k: 1.0 if want & set(ranked[:k]) else 0.0 for k in TOP_K})
    return out


def main() -> None:
    gold, _ = gold_and_turns()
    memos = Path(os.environ["MEMOS_NOTES"])
    asked = [
        (row["question"], gold[row["question"]])
        for row in json.loads(memos.read_text(encoding="utf-8"))
        if row["question"] in gold
    ]

    print(f"同一批 {len(asked)} 题（MemOS/Cognee 实际回答过的那些）\n")
    print(f"{'方案':<34}" + "".join(f"{'recall@' + str(k):>11}" for k in TOP_K))
    print("-" * 78)

    rows = ours_on(asked)
    cells = "".join(f"{statistics.mean(r[k] for r in rows) * 100:>10.1f}%" for k in TOP_K)
    print(f"{'OpenKimo（同题集重算）':<32}{cells}")

    score(memos, "MemOS（返回 8 条原文轮次）", per_result=1)
    score(Path(os.environ["COGNEE_NOTES"]), "Cognee 取回原文块", per_result=20)


if __name__ == "__main__":
    main()
