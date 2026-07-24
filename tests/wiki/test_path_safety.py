from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from kimi_cli.wiki.models import CurrentSource, SourceRef, UnsafeWikiPath
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


@pytest.mark.parametrize(
    "unsafe_path",
    [
        r"C:\Users\person\source.md",
        "C:/Users/person/source.md",
        r"\\server\share\source.md",
        "//server/share/source.md",
    ],
)
def test_source_models_reject_windows_drive_and_unc_paths(unsafe_path: str) -> None:
    with pytest.raises(ValueError):
        SourceRef(
            kind="workspace-file",
            workspace_id="123e4567-e89b-12d3-a456-426614174000",
            path=unsafe_path,
            content_hash="sha256:" + "a" * 64,
        )
    with pytest.raises(ValueError):
        CurrentSource(
            kind="workspace-file",
            workspace_id="123e4567-e89b-12d3-a456-426614174000",
            relative_path=unsafe_path,
        )
