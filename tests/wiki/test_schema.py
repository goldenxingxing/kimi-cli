from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from kimi_cli.wiki.models import SourceRef, UnsafeWikiPage, WikiPage
from kimi_cli.wiki.schema import content_hash, parse_page, render_page

VALID_PAGE = """---
title: 原子写入
created: 2026-07-24T12:00:00+08:00
updated: 2026-07-24T12:00:00+08:00
tags: [并发, safety]
sources:
  - kind: workspace-file
    workspace_id: 123e4567-e89b-12d3-a456-426614174000
    path: docs/atomic-writes.md
    content_hash: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
revision: 1
---
使用 [[concepts/atomic-writes]] 保证写入完整。
"""


def test_page_round_trip_increments_revision() -> None:
    page = parse_page(VALID_PAGE, "concepts/atomic-writes.md")
    updated = page.model_copy(update={"revision": page.revision + 1})

    assert parse_page(render_page(updated), "concepts/atomic-writes.md").revision == 2
    assert page.title == "原子写入"
    assert page.created == datetime.fromisoformat("2026-07-24T12:00:00+08:00")
    assert page.sources[0].workspace_id == UUID("123e4567-e89b-12d3-a456-426614174000")


def test_content_hash_is_sha256_prefixed_and_deterministic() -> None:
    assert (
        content_hash(b"wiki")
        == "sha256:" + "12a435ec8454c6d1c90a1d92812b09af11bee711fbe524d56a8f26ea7c5ccee8"
    )


@pytest.mark.parametrize(
    "text",
    [
        "# missing frontmatter\n",
        "---\ntitle: missing fields\n---\nbody\n",
        "---\ntitle: bad\ncreated: no\n---\nbody\n",
        VALID_PAGE.replace("revision: 1", "revision: 0"),
        VALID_PAGE.replace("kind: workspace-file", "kind: archive"),
        VALID_PAGE.replace("[[concepts/atomic-writes]]", "[[comparations/typo]]"),
        VALID_PAGE.replace("[[concepts/atomic-writes]]", "[[concepts/atomic-writes"),
        VALID_PAGE.replace("使用", "api_key: sk-abcdefghijklmnopqrstuvwxyz0123456789"),
        VALID_PAGE.replace("使用", "记录在 /Users/person/private/research.md。"),
        VALID_PAGE.replace("使用", "记录在 /opt/openkimo/wiki.md。"),
        VALID_PAGE.replace("使用", "记录在 C:/Users/person/wiki.md。"),
        VALID_PAGE.replace("使用", r"记录在 C:\Users\person\wiki.md。"),
        VALID_PAGE.replace("使用", r"记录在 \\server\share\wiki.md。"),
        VALID_PAGE.replace("使用", "记录在 //server/share/wiki.md。"),
    ],
)
def test_page_rejects_malformed_or_unsafe_content(text: str) -> None:
    with pytest.raises((UnsafeWikiPage, ValidationError, ValueError)):
        parse_page(text, "concepts/atomic-writes.md")


def test_page_rejects_absolute_or_sensitive_provenance() -> None:
    absolute = VALID_PAGE.replace("path: docs/atomic-writes.md", "path: /Users/person/private.md")
    sensitive = VALID_PAGE.replace("path: docs/atomic-writes.md", "path: .aws/credentials")

    for text in (absolute, sensitive):
        with pytest.raises((UnsafeWikiPage, ValidationError, ValueError)):
            parse_page(text, "concepts/atomic-writes.md")


def test_page_allows_normal_https_url_in_markdown_body() -> None:
    text = VALID_PAGE.replace(
        "使用", "参考 [公开资料](https://example.test/docs/wiki?topic=api_key)，并"
    )

    assert parse_page(text, "concepts/atomic-writes.md").title == "原子写入"


def test_web_source_rejects_credential_bearing_url() -> None:
    with pytest.raises(ValidationError):
        SourceRef(
            kind="web",
            url="https://user:password@example.test/source",
            content_hash="sha256:" + "a" * 64,
        )


@pytest.mark.parametrize(
    "query_name",
    [
        "apiKey",
        "api%4bey",
        "credential",
        "cookie",
        "sessionid",
        "X-Amz-Signature",
        "authorization",
        "token",
        "password",
    ],
)
def test_web_source_rejects_normalized_secret_query_parameter(query_name: str) -> None:
    with pytest.raises(ValidationError):
        SourceRef(
            kind="web",
            url=f"https://example.test/source?{query_name}=secret-value",
            content_hash="sha256:" + "a" * 64,
        )


def test_direct_page_model_rejects_unsafe_logical_path() -> None:
    page = parse_page(VALID_PAGE, "concepts/atomic-writes.md")

    with pytest.raises(ValidationError):
        WikiPage(**(page.model_dump() | {"logical_path": "../secret.md"}))
