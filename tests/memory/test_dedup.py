"""One constraint this application places on the memory package.

The matching rules themselves are tested in Amem, where they live. What cannot
be tested there is this: the Wiki's intent hashing is an authorization
boundary, and it folds strings with the same function memory uses to decide
whether two statements are the same. The two must not drift apart, and they
now sit in different repositories, so the assertion belongs here.
"""

from __future__ import annotations

import pytest
from amem import fold_text

from kimi_cli.wiki.intent import normalize_intent_text


@pytest.mark.parametrize(
    "raw",
    ["Ｒemember  THIS\n rule", "记住 A/B", "plain text", "", "   "],
)
def test_fold_text_still_backs_the_wiki_normalizer(raw: str) -> None:
    # normalize_intent_text is an authorization boundary; extracting the shared
    # helper must not have changed what it folds.
    assert normalize_intent_text(raw) == fold_text(raw)
