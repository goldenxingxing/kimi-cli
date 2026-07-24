from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from kimi_cli.wiki.models import UnsafeWikiPath
from kimi_cli.wiki.schema import resolve_page_path, validate_logical_page


@pytest.mark.parametrize(
    "bad",
    ["../secret.md", "/tmp/x.md", "entities/../../x.md", ".openkimo/revision", "schema.md"],
)
def test_logical_page_rejects_escape(bad: str) -> None:
    with pytest.raises(UnsafeWikiPath):
        validate_logical_page(bad)


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        ("entities/openkimo.md", PurePosixPath("entities/openkimo.md")),
        ("concepts/中文.md", PurePosixPath("concepts/中文.md")),
        ("comparisons/file-locks.md", PurePosixPath("comparisons/file-locks.md")),
    ],
)
def test_logical_page_accepts_declared_category_and_slug(
    page: str, expected: PurePosixPath
) -> None:
    assert validate_logical_page(page) == expected


def test_resolved_page_path_stays_below_wiki_root(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    (root / "concepts").mkdir(parents=True)

    assert (
        resolve_page_path(root, "concepts/atomic-writes.md") == root / "concepts/atomic-writes.md"
    )


def test_resolved_page_path_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "concepts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeWikiPath):
        resolve_page_path(root, "concepts/atomic-writes.md")
