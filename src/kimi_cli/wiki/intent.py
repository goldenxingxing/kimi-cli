"""Bounded detection of explicitly stated durable-knowledge intent.

The detector reads only text the user actually authored and accepted in a real
turn.  It answers one narrow question — did the user *say* this should outlive
the session — and never inspects tool output, model text, or Wiki content.  A
match is a trigger to ask, not authority to write: admission still requires a
runtime-created checkpoint and grant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from kimi_cli.utils.string import fold_text
from kimi_cli.wiki.schema import content_hash

DurableIntentFamily = Literal["remember", "preference", "fixed_rule"]

MAX_INTENT_BYTES = 4 * 1024
"""Only this much user text is scanned; durable intent is stated up front."""

_FAMILY_ORDER: tuple[DurableIntentFamily, ...] = ("remember", "preference", "fixed_rule")

# Negations and single-run scoping are checked first and always win.  "Do not
# remember this" contains a perfectly good positive marker.
_NEGATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:do\s*n[o']?t|don't|never|no\s+need\s+to|stop|avoid|without)\s+"
        r"(?:ever\s+)?(?:remember|memoriz|memoris|record|save|store|persist|"
        r"keep|note|write\s+down|write\s+to|add\s+to)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:only|just)\s+(?:for\s+)?(?:this|the\s+current|one)\s+"
        r"(?:run|turn|time|session|conversation|chat|task|once)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:for\s+)?(?:this\s+(?:run|turn|time|session|conversation|chat|task)\s+only|"
        r"one[-\s]?off|temporar(?:y|ily)|throwaway|scratch|ad[-\s]?hoc)\b",
        re.IGNORECASE,
    ),
    re.compile(r"不要|别|无需|不用|不必|勿|禁止|不需要"),
    re.compile(
        r"(?:仅|只|就)(?:限|是|)(?:这|此|本)(?:次|回|轮|一次)|这次(?:临时|先|就)|临时(?:使用|用一下|用)?"
    ),
)

_FAMILY_PATTERNS: dict[DurableIntentFamily, tuple[re.Pattern[str], ...]] = {
    "remember": (
        re.compile(
            r"\b(?:remember|memoriz\w*|memoris\w*|keep\s+in\s+mind|make\s+a\s+note|note\s+down|"
            r"write\s+(?:this\s+)?down|save|record|store|persist)\b[^.!?\n]{0,80}?"
            r"\b(?:this|that|it|for\s+(?:future|later|next)|permanently|forever|"
            r"across\s+sessions?|long[-\s]?term)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:remember|note|save|record|store)\s+(?:this|that|the\s+following)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\badd\s+(?:this|that|it)\s+to\s+(?:the\s+)?(?:wiki|knowledge\s*base)\b", re.IGNORECASE
        ),
        re.compile(r"(?:请\s*)?(?:记住|记下|牢记|记录下来|存(?:入|到)|写入)(?:知识库|文档|笔记)?"),
        re.compile(r"(?:加入|添加到|收录到)\s*(?:全局)?知识库"),
    ),
    "preference": (
        re.compile(
            r"\bfrom\s+now\s+on\b|\bgoing\s+forward\b|\bin\s+(?:the\s+)?future\b|"
            r"\bfor\s+(?:all\s+)?future\s+(?:sessions?|runs?|tasks?|work)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bI\s+(?:always\s+)?prefer\b|\bmy\s+preference\s+is\b|\bI'?d\s+always\s+like\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"以后(?:都|请|要|所有|一律)?|今后|从(?:现在|此)(?:开始|起)|默认(?:都|使用|采用)"
        ),
        re.compile(r"我(?:更)?(?:喜欢|偏好|习惯)|我的偏好是"),
    ),
    "fixed_rule": (
        re.compile(
            r"\b(?:permanent|standing|fixed|invariant|non[-\s]?negotiable)\s+"
            r"(?:rule|policy|convention|standard|requirement)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?rule\s+is\b|\bour\s+(?:policy|convention|standard)\s+is\b|"
            r"\bmust\s+always\b|\balways\s+must\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:固定|长期|永久|既定)(?:规则|规范|约定|要求|策略)|规矩是|规则是"),
        re.compile(
            r"(?:后续|所有|每个|全部)(?:的)?(?:工作区|项目|仓库|会话)(?:都)?(?:要|必须|需)(?:遵守|遵循|执行)"
        ),
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class DurableIntent:
    """One detected statement that knowledge should outlive this session."""

    family: DurableIntentFamily
    raw_hash: str
    normalized_hash: str


def normalize_intent_text(raw_text: str) -> str:
    """Fold Unicode width, whitespace, and case without dropping any content.

    Only these three axes are normalized: the normalized hash must still stand
    for the same statement the user made, so a later turn cannot pass off
    different text as the same accepted intent. Any additional folding belongs
    to the caller, not here — this is an authorization boundary, and a fold
    added for someone else's convenience widens what counts as the same intent.
    """
    return fold_text(raw_text)


def detect_durable_intent(raw_text: str) -> DurableIntent | None:
    """Return the single durable intent stated in *raw_text*, if any.

    Negated and single-run phrasings are rejected before any positive marker is
    considered, and at most one family is returned, in the fixed precedence
    order ``remember``, ``preference``, ``fixed_rule``.
    """
    if not raw_text or not raw_text.strip():
        return None
    scanned = _bounded(raw_text)
    normalized = normalize_intent_text(scanned)
    if not normalized:
        return None
    if any(pattern.search(normalized) for pattern in _NEGATIVE_PATTERNS):
        return None
    for family in _FAMILY_ORDER:
        if any(pattern.search(normalized) for pattern in _FAMILY_PATTERNS[family]):
            return DurableIntent(
                family=family,
                raw_hash=content_hash(raw_text.encode("utf-8")),
                normalized_hash=content_hash(normalized.encode("utf-8")),
            )
    return None


def _bounded(raw_text: str) -> str:
    encoded = raw_text.encode("utf-8")
    if len(encoded) <= MAX_INTENT_BYTES:
        return raw_text
    return encoded[:MAX_INTENT_BYTES].decode("utf-8", errors="ignore")
