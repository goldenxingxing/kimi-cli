"""End-to-end question answering, judged by a model.

Recall of the gold evidence turn is a fair metric only for systems that store
turns. A memory system whose value is synthesis — a graph that knows Caroline
joined a support group in May because three separate turns implied it — cannot
score on it at all: the answer it returns is correct and corresponds to no
single turn. Scoring that as a miss measures the metric, not the system.

So the question here is the one a user actually has: given what the system
retrieved, can the question be answered correctly? Every system gets the same
question, returns whatever it returns, and a model that is not told which
system produced what marks the answer against the gold. That is the only shape
in which a turn store and a knowledge graph are comparable.

The judge is the weak point and is treated as one: it sees only the gold answer
and the candidate, never the system's name, and is told to accept a different
wording of the same fact and reject a plausible fact that is not the gold one.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

import httpx

_ANSWER_PROMPT = """\
Answer the question using only the notes below. If the notes do not contain \
the answer, reply exactly: UNKNOWN.

Answer in as few words as possible — a name, a date, a phrase. No explanation.

NOTES:
{notes}

QUESTION: {question}
ANSWER:"""

_JUDGE_PROMPT = """\
You are marking one answer against a reference.

QUESTION: {question}
REFERENCE ANSWER: {gold}
GIVEN ANSWER: {given}

Mark CORRECT if the given answer states the same fact as the reference, even \
in different words, a different language, or with more or less detail. \
Mark WRONG if it states a different fact, hedges without committing, or says \
it does not know. A plausible answer that is not the reference is WRONG.

Reply with exactly one word: CORRECT or WRONG."""


@dataclass(frozen=True, slots=True)
class Judged:
    question: str
    gold: str
    given: str
    correct: bool


class Model:
    """One OpenAI-compatible endpoint, with a cap on concurrency."""

    def __init__(self, concurrency: int = 8, *, alt: bool = False) -> None:
        """*alt* answers with the second endpoint instead of the default one.

        A system that answers for itself is using its own model, so its score
        mixes the memory with the model. The only way to tell those apart is to
        run somebody else's notes through the same model — hence this switch.
        """
        self._sem = asyncio.Semaphore(concurrency)
        prefix = "GRAPH" if alt else "BENCH"
        self._url = os.environ[f"{prefix}_BASE_URL"].rstrip("/") + "/chat/completions"
        self._key = os.environ["GRAPH_API_KEY" if alt else "DEEPSEEK_API_KEY"]
        self._model = os.environ[f"{prefix}_MODEL"]
        # k3 refuses `temperature: 0` outright — "only 1 is allowed for this
        # model" — and the failure arrives as an empty answer, which scores as
        # a wrong one. A control group that reads 0.0% is a broken call, not a
        # result, and this is the second time in this benchmark that an
        # implausible number was the only thing that gave it away.
        self._fixed_temperature = not alt

    async def ask(self, client: httpx.AsyncClient, prompt: str, *, retries: int = 3) -> str:
        async with self._sem:
            for attempt in range(retries):
                try:
                    r = await client.post(
                        self._url,
                        headers={"Authorization": f"Bearer {self._key}"},
                        json={
                            "model": self._model,
                            "messages": [{"role": "user", "content": prompt}],
                            **({"temperature": 0} if self._fixed_temperature else {}),
                        },
                    )
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"].strip()
                except Exception:
                    if attempt == retries - 1:
                        return ""
                    await asyncio.sleep(1.5 * (attempt + 1))
        return ""


async def answer_and_judge(
    model: Model,
    client: httpx.AsyncClient,
    question: str,
    gold: str,
    notes: str,
) -> Judged:
    given = await model.ask(
        client, _ANSWER_PROMPT.format(notes=notes or "(none)", question=question)
    )
    if not given or given.strip().upper().startswith("UNKNOWN"):
        return Judged(question, gold, given or "(no answer)", correct=False)
    verdict = await model.ask(
        client, _JUDGE_PROMPT.format(question=question, gold=gold, given=given)
    )
    return Judged(question, gold, given, correct=bool(re.match(r"\s*CORRECT", verdict, re.I)))


async def score_all(
    retrieved: dict[str, list[tuple[str, str, str]]],
    concurrency: int = 8,
    *,
    alt: bool = False,
) -> dict[str, list[Judged]]:
    """`{system: [(question, gold, notes), ...]}` -> the same keys, judged.

    Answering and judging run together across systems so a slow retriever does
    not serialise the rest.
    """
    model = Model(concurrency, alt=alt)
    out: dict[str, list[Judged]] = {}
    async with httpx.AsyncClient(timeout=180) as client:
        tasks = {
            name: [answer_and_judge(model, client, q, g, n) for q, g, n in rows]
            for name, rows in retrieved.items()
        }
        for name, coros in tasks.items():
            out[name] = list(await asyncio.gather(*coros))
    return out
