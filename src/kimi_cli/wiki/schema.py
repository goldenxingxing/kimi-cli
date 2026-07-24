"""Parse, render, and safely locate authoritative global Wiki pages."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml
from pydantic import ValidationError

from kimi_cli.wiki.models import UnsafeWikiPage, UnsafeWikiPath, WikiPage

_CATEGORIES = frozenset({"entities", "concepts", "comparisons", "sources", "queries", "lint"})
_SLUG_RE = re.compile(r"[\w][\w-]*", flags=re.UNICODE)
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?im)^\s*(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[^\s]{12,}$"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"),
)
_MACHINE_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?i)(?<![\w-])file:(?://)?[^\s\])}>]*"),
    re.compile(
        r"(?<![\w:/])/(?:Applications|Library|System|Users|Volumes|bin|dev|etc|home|mnt|opt|private|proc|root|run|sbin|tmp|usr|var)(?:/|(?=\s|[\])}>.,;!?]|$))"
    ),
    re.compile(r"(?<!\w)[A-Za-z]:[\\/]+[^\s\])}>]*"),
    re.compile(r"(?<![\\/:])\\\\+[^\s\])}>]+"),
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
    if any(pattern.search(body) for pattern in _MACHINE_ABSOLUTE_PATH_PATTERNS):
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
