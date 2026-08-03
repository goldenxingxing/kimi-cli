"""Tests for bounded, negation-aware durable-intent detection."""

from __future__ import annotations

import pytest

from kimi_cli.wiki.intent import (
    MAX_INTENT_BYTES,
    detect_durable_intent,
    normalize_intent_text,
)
from kimi_cli.wiki.schema import content_hash


@pytest.mark.parametrize(
    ("text", "family"),
    [
        ("Remember this release rule for future sessions.", "remember"),
        ("Please note this down permanently.", "remember"),
        ("Add this to the knowledge base.", "remember"),
        ("写入知识库：发布前必须运行完整测试。", "remember"),
        ("请记住这个发布规则。", "remember"),
        ("From now on, I prefer concise reports.", "preference"),
        ("Going forward, use tabs.", "preference"),
        ("以后都使用英文提交信息。", "preference"),
        ("我更喜欢简洁的报告。", "preference"),
        ("Our permanent release rule is signed tags only.", "fixed_rule"),
        ("The rule is: signed tags only.", "fixed_rule"),
        ("后续所有工作区都要遵守这个规则。", "fixed_rule"),
        ("这是长期规范。", "fixed_rule"),
    ],
)
def test_durable_intent_positive_patterns(text: str, family: str) -> None:
    result = detect_durable_intent(text)

    assert result is not None
    assert result.family == family


@pytest.mark.parametrize(
    "text",
    [
        "Do not remember this",
        "Don't save this anywhere",
        "Never record this preference",
        "only for this run",
        "just for this session, prefer concise reports",
        "Remember this, but only for this run",
        "This is a temporary workaround; from now on ignore it",
        "不要记住",
        "无需写入知识库",
        "这次临时使用",
        "仅限这次，以后不用",
        "build is running",
        "",
        "   ",
    ],
)
def test_durable_intent_negation_and_temporary_forms_do_not_match(text: str) -> None:
    assert detect_durable_intent(text) is None


def test_durable_intent_precedence_is_remember_then_preference_then_fixed_rule() -> None:
    text = "From now on our permanent rule is X. Please remember this."

    result = detect_durable_intent(text)

    assert result is not None
    assert result.family == "remember"


def test_durable_intent_keeps_raw_and_normalized_hashes_separate() -> None:
    raw = "  Remember   This   Release Rule  "

    result = detect_durable_intent(raw)

    assert result is not None
    assert result.raw_hash == content_hash(raw.encode("utf-8"))
    assert result.normalized_hash == content_hash(normalize_intent_text(raw).encode("utf-8"))
    assert result.raw_hash != result.normalized_hash


def test_normalization_folds_only_width_whitespace_and_case() -> None:
    assert normalize_intent_text("Ｒemember  THIS\n rule") == "remember this rule"
    # Content itself is never dropped.
    assert normalize_intent_text("记住 A/B") == "记住 a/b"


def test_two_spellings_of_the_same_statement_share_a_normalized_hash() -> None:
    left = detect_durable_intent("Remember this rule")
    right = detect_durable_intent("REMEMBER   this\trule")

    assert left is not None and right is not None
    assert left.normalized_hash == right.normalized_hash
    assert left.raw_hash != right.raw_hash


def test_detection_scans_only_a_bounded_prefix_of_user_text() -> None:
    filler = "a" * MAX_INTENT_BYTES
    beyond_budget = f"{filler} please remember this rule"

    assert detect_durable_intent(beyond_budget) is None
    assert detect_durable_intent("please remember this rule") is not None


def test_multibyte_truncation_never_splits_a_code_point() -> None:
    # Chinese runs 3 bytes per character, so the budget lands mid-character.
    filler = "测" * (MAX_INTENT_BYTES // 3)
    assert detect_durable_intent(f"{filler}请记住这个规则") is None
