"""Read-only health checks for authoritative Wiki Markdown."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from kimi_cli.wiki.models import WikiPage
from kimi_cli.wiki.schema import content_hash, parse_page

LintCode = Literal[
    "malformed_page",
    "dead_link",
    "orphan",
    "duplicate_hash",
    "duplicate_claim",
    "conflict_marker",
    "missing_provenance",
]
_WIKILINK = re.compile(r"\[\[([^\[\]]+)\]\]")
_CONFLICT = re.compile(r"(?im)^##+\s+Conflict\b|^(?:<{7}|={7}|>{7})")
_CLAIM = re.compile(r"(?m)^(?!\s*#)(?:\s*[-*]\s+)?(.{24,})$")


@dataclass(frozen=True, slots=True)
class LintIssue:
    code: LintCode
    logical_path: str
    detail: str
    related_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LintReport:
    scope: str | None
    scanned_pages: int
    modified_pages: int
    issues: tuple[LintIssue, ...]


def lint_snapshot(raw_pages: Mapping[str, str], *, scope: str | None) -> LintReport:
    """Inspect one committed raw Markdown snapshot without mutating it."""
    selected = {
        path: text
        for path, text in raw_pages.items()
        if scope is None or path.startswith(f"{scope}/")
    }
    issues: list[LintIssue] = []
    all_pages: dict[str, WikiPage] = {}
    parse_errors: dict[str, str] = {}
    for logical_path, text in sorted(raw_pages.items()):
        try:
            all_pages[logical_path] = parse_page(text, logical_path)
        except (ValueError, UnicodeError) as exc:
            parse_errors[logical_path] = str(exc)
    pages = {path: all_pages[path] for path in selected if path in all_pages}
    for logical_path in sorted(set(selected) - set(pages)):
        detail = parse_errors.get(logical_path, "page cannot be parsed")
        issues.append(
            LintIssue(
                code="malformed_page",
                logical_path=logical_path,
                detail=detail,
            )
        )

    valid_paths = set(all_pages)
    inbound: dict[str, set[str]] = defaultdict(set)
    for source_path, page in all_pages.items():
        for target in _WIKILINK.findall(page.body):
            target_path = f"{target}.md"
            if target_path in valid_paths:
                inbound[target_path].add(source_path)

    body_hashes: dict[str, list[str]] = defaultdict(list)
    claims: dict[str, list[str]] = defaultdict(list)
    for logical_path, page in sorted(pages.items()):
        for target in _WIKILINK.findall(page.body):
            target_path = f"{target}.md"
            if target_path not in valid_paths:
                issues.append(
                    LintIssue(
                        code="dead_link",
                        logical_path=logical_path,
                        detail=f"missing target: {target_path}",
                        related_paths=(target_path,),
                    )
                )
        if not page.sources:
            issues.append(
                LintIssue(
                    code="missing_provenance",
                    logical_path=logical_path,
                    detail="page has no source provenance",
                )
            )
        if _CONFLICT.search(page.body):
            issues.append(
                LintIssue(
                    code="conflict_marker",
                    logical_path=logical_path,
                    detail="page contains an explicit unresolved conflict",
                )
            )
        body_hashes[content_hash(page.body.strip().encode("utf-8"))].append(logical_path)
        for claim in _CLAIM.findall(page.body):
            normalized = " ".join(claim.casefold().split())
            if normalized:
                claims[normalized].append(logical_path)

    for logical_path in sorted(pages):
        if not inbound.get(logical_path):
            issues.append(
                LintIssue(
                    code="orphan",
                    logical_path=logical_path,
                    detail="page has no inbound Wiki links",
                )
            )
    _append_duplicates(issues, body_hashes, code="duplicate_hash")
    _append_duplicates(issues, claims, code="duplicate_claim")
    return LintReport(
        scope=scope,
        scanned_pages=len(selected),
        modified_pages=0,
        issues=tuple(
            sorted(issues, key=lambda issue: (issue.logical_path, issue.code, issue.detail))
        ),
    )


def _append_duplicates(
    issues: list[LintIssue],
    groups: Mapping[str, list[str]],
    *,
    code: Literal["duplicate_hash", "duplicate_claim"],
) -> None:
    for paths in groups.values():
        unique = tuple(sorted(set(paths)))
        if len(unique) < 2:
            continue
        for logical_path in unique:
            issues.append(
                LintIssue(
                    code=code,
                    logical_path=logical_path,
                    detail=f"duplicate content also appears in {len(unique) - 1} page(s)",
                    related_paths=tuple(path for path in unique if path != logical_path),
                )
            )
