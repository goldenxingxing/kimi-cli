"""Parse, render, and safely locate authoritative global Wiki pages."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import unquote

import yaml
from pydantic import ValidationError

from kimi_cli.wiki.models import (
    UnsafeWikiPage,
    UnsafeWikiPath,
    WikiPage,
    has_sensitive_url_parameters,
)

_CATEGORIES = frozenset({"entities", "concepts", "comparisons", "sources", "queries", "lint"})
_SLUG_RE = re.compile(r"[\w][\w-]*", flags=re.UNICODE)
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?im)^\s*(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[^\s]{12,}$"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"),
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)^\s*(?:access[_-]?token|api[_-]?key|auth[_-]?token|client[_-]?secret|cookie|id[_-]?token|private[_-]?key|refresh[_-]?token|secret[_-]?key|session[_-]?token|user[_-]?password|user[_-]?token)\s*[:=]\s*\S+"
)
_SECRET_HEADER_RE = re.compile(r"(?im)^\s*(?:authorization|cookie|set-cookie)\s*:\s*\S+")
_FILE_URI_RE = re.compile(r"(?i)(?<![\w:/-])file:(?://|[\\/])")
_ROOT_RELATIVE_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*/(?!/)[^\s)]+\s*\)")
_API_ENDPOINT_RE = re.compile(r"(?<![\w:/])/api(?:/[^\s\])}>,;!?]+)*")
_BODY_HTTP_URL_RE = re.compile(r"https?://[^\s\])}>]+", flags=re.IGNORECASE)
_MARKDOWN_URL_TARGET_RE = re.compile(r"!?\[[^\]]*\]\(\s*([^\s)]+)")
_MACHINE_ABSOLUTE_PATH_PATTERNS = (
    # After safe Markdown links and API endpoints are removed, a leading slash is
    # unambiguously a local POSIX path rather than a root-relative web route.
    re.compile(r"(?<![\w:/.])/(?!/)[^\s\])}>.,;!?]+"),
    re.compile(r"(?<!\w)[A-Za-z]:[\\/]+[^\s\])}>]*"),
    re.compile(r"(?<![\\/:])\\\\+[^\s\])}>]+"),
    re.compile(r"(?<!\\)\\(?!\\)(?=[A-Za-z0-9_.-]+(?:[\\/]|$))"),
    re.compile(r"(?<![/:])//[^\s\])}>]+"),
)
_FRONTMATTER_FIELDS = frozenset({"title", "created", "updated", "tags", "sources", "revision"})


def content_hash(data: bytes) -> str:
    """Return the portable SHA-256 identifier used by source provenance."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def validate_logical_page(page: str) -> PurePosixPath:
    """Return a canonical content-page path or reject paths outside the schema."""
    path = PurePosixPath(page)
    if (
        not page
        or "\\" in page
        or path.is_absolute()
        or len(path.parts) != 2
        or ".." in path.parts
        or "." in path.parts
        or path.suffix != ".md"
        or path.parts[0] not in _CATEGORIES
        or path.as_posix() != page
        or not _SLUG_RE.fullmatch(path.stem)
    ):
        raise UnsafeWikiPath(page)
    return path


def resolve_page_path(root: Path, logical_path: str) -> Path:
    """Resolve a logical path and reject final or intermediate symlink escapes."""
    relative = validate_logical_page(logical_path)
    managed_root = root.resolve(strict=False)
    target = (managed_root / relative).resolve(strict=False)
    if not target.is_relative_to(managed_root):
        raise UnsafeWikiPath(logical_path)
    return target


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise UnsafeWikiPage("Wiki pages must start with YAML frontmatter")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            raw = "".join(lines[1:index])
            body = "".join(lines[index + 1 :])
            try:
                parsed = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise UnsafeWikiPage("invalid Wiki YAML frontmatter") from exc
            if not isinstance(parsed, dict):
                raise UnsafeWikiPage("Wiki frontmatter must be a mapping")
            frontmatter = cast(dict[str, Any], parsed)
            if frozenset(frontmatter) != _FRONTMATTER_FIELDS:
                raise UnsafeWikiPage("Wiki frontmatter fields must exactly match the page schema")
            return frontmatter, body
    raise UnsafeWikiPage("Wiki frontmatter is missing its closing delimiter")


def _validate_body(body: str) -> None:
    if not body.strip():
        raise UnsafeWikiPage("Wiki page body cannot be empty")
    if "[[" in body or "]]" in body:
        links = list(_WIKILINK_RE.finditer(body))
        consumed = _WIKILINK_RE.sub("", body)
        if "[[" in consumed or "]]" in consumed:
            raise UnsafeWikiPage("Wiki links must use [[category/slug]]")
        for link in links:
            validate_logical_page(f"{link.group(1)}.md")
    if any(pattern.search(body) for pattern in _SECRET_PATTERNS):
        raise UnsafeWikiPage("Wiki pages cannot contain credentials or secrets")
    if _SECRET_ASSIGNMENT_RE.search(body) or _SECRET_HEADER_RE.search(body):
        raise UnsafeWikiPage("Wiki pages cannot contain credential assignments or headers")
    url_candidates = _BODY_HTTP_URL_RE.findall(body) + _MARKDOWN_URL_TARGET_RE.findall(body)
    if any(has_sensitive_url_parameters(url) for url in url_candidates):
        raise UnsafeWikiPage("Wiki page URLs cannot contain credential parameters")
    if _FILE_URI_RE.search(body):
        raise UnsafeWikiPage("Wiki pages cannot contain local file URIs")
    api_endpoints = tuple(_API_ENDPOINT_RE.finditer(body))
    if any(
        any(segment in {".", ".."} for segment in unquote(match.group()).split("/"))
        for match in api_endpoints
    ):
        raise UnsafeWikiPage("Wiki pages cannot contain traversal segments in API paths")
    path_context = _ROOT_RELATIVE_MARKDOWN_LINK_RE.sub("", body)
    path_context = _API_ENDPOINT_RE.sub("", path_context)
    if any(pattern.search(path_context) for pattern in _MACHINE_ABSOLUTE_PATH_PATTERNS):
        raise UnsafeWikiPage("Wiki pages cannot contain machine-specific absolute paths")


def parse_page(text: str, logical_path: str) -> WikiPage:
    """Parse one strict Markdown content page from the authoritative store."""
    path = validate_logical_page(logical_path)
    frontmatter, body = _split_frontmatter(text)
    _validate_body(body)
    try:
        return WikiPage(logical_path=path.as_posix(), body=body, **frontmatter)
    except ValidationError as exc:
        raise UnsafeWikiPage("Wiki page does not satisfy the required schema") from exc


def render_page(page: WikiPage) -> str:
    """Render a validated page with deterministic field ordering and UTF-8-safe YAML."""
    validate_logical_page(page.logical_path)
    _validate_body(page.body)
    frontmatter = {
        "title": page.title,
        "created": page.created.isoformat(),
        "updated": page.updated.isoformat(),
        "tags": page.tags,
        "sources": [source.model_dump(mode="json", exclude_none=True) for source in page.sources],
        "revision": page.revision,
    }
    yaml_text = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{yaml_text}\n---\n{page.body}"
