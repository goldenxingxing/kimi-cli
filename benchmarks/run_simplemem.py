"""SimpleMem over LoCoMo.

Its pitch is compression-first: structure a conversation into dense memory,
then answer from that rather than from passages. Like Cognee it answers for
itself (`ask`), so its output is judged directly rather than fed to a shared
answering model — the same asymmetry, labelled the same way.

It takes a timestamp per turn, which matters here: nearly a quarter of LoCoMo's
answers are dates and the date lives on the session, not in the text.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# DeepSeek rather than k3: SimpleMem sets a temperature on every call and k3
# accepts only 1 ("invalid temperature: only 1 is allowed for this model"),
# which fails every request and leaves an empty store behind. SimpleMem parses
# the model's text itself, so it does not need the structured output that sent
# the graph builders to k3 in the first place.
os.environ.setdefault("OPENAI_API_KEY", os.environ["DEEPSEEK_API_KEY"])
os.environ.setdefault("OPENAI_BASE_URL", os.environ["BENCH_BASE_URL"].rstrip("/"))
os.environ.setdefault("LLM_MODEL", os.environ["BENCH_MODEL"])
os.environ.setdefault("SIMPLEMEM_MODEL", os.environ["BENCH_MODEL"])
# Loaded locally through sentence-transformers rather than over an embeddings
# API, so this has to be a real Hugging Face repo id: the short name resolves
# to `sentence-transformers/bge-small-en-v1.5`, which does not exist, and the
# failure arrives as a 401 that reads like an auth problem.
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("EMBEDDING_DIMENSION", "384")
os.environ.setdefault("SIMPLEMEM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

import simplemem  # noqa: E402  (must follow the environment setup above)

DATA = Path(os.environ["BENCH_DATA_PATH"])
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "2"))
QUESTIONS = int(os.environ.get("BENCH_QUESTIONS", "60"))


def sessions(sample):
    convo = sample["conversation"]
    names = sorted(
        (k for k in convo if k.startswith("session_") and "date" not in k),
        key=lambda n: int(n.split("_")[1]),
    )
    for name in names:
        yield convo.get(f"{name}_date_time", ""), convo[name]


def main() -> None:
    samples = json.loads(DATA.read_text(encoding="utf-8"))[:SAMPLES]
    out = []
    for index, sample in enumerate(samples):
        os.environ["LANCEDB_PATH"] = f"/tmp/simplemem-{index}"
        system = simplemem.create_system(clear_db=True)

        added = 0
        for when, turns in sessions(sample):
            for turn in turns:
                if not turn.get("text"):
                    continue
                try:
                    system.add_dialogue(
                        speaker=turn.get("speaker", "?"),
                        content=turn["text"],
                        timestamp=when or None,
                    )
                    added += 1
                except Exception as exc:
                    print(f"  add 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
                    break
        print(f"  样本 {index + 1}: 灌入 {added} 轮，整理中…", flush=True)
        try:
            system.finalize()
        except Exception as exc:
            print(f"  finalize 失败: {type(exc).__name__}: {exc}", file=sys.stderr)

        asked = 0
        for qa in sample.get("qa", []):
            q, a = (qa.get("question") or "").strip(), str(qa.get("answer") or "").strip()
            if not q or not a or asked >= QUESTIONS:
                continue
            try:
                answer = system.ask(q) or ""
            except Exception as exc:
                answer = ""
                print(f"  ask 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            out.append({"question": q, "gold": a, "graph_answer": answer, "notes": ""})
            asked += 1
            if asked % 20 == 0:
                print(f"    已问 {asked} 题", flush=True)
        print(f"  样本 {index + 1} 完成：{asked} 题", flush=True)

    Path(os.environ["BENCH_OUT"]).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"写出 {len(out)} 条")


if __name__ == "__main__":
    main()
