"""Tests for the write_file tool."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from kaos.path import KaosPath
from pydantic import ValidationError

from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval, ApprovalResult
from kimi_cli.tools.file.write import Params, WriteFile
from kimi_cli.wiki.manager import WikiManager
from kimi_cli.wire.types import DiffDisplayBlock


async def test_write_new_file(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing a new file."""
    file_path = temp_work_dir / "new_file.txt"
    content = "Hello, World!"

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert "successfully overwritten" in result.message
    diff_block = next(block for block in result.display if block.type == "diff")
    assert isinstance(diff_block, DiffDisplayBlock)
    assert diff_block.path == str(file_path)
    assert diff_block.old_text == ""
    assert diff_block.new_text == content
    assert await file_path.exists()
    assert await file_path.read_text() == content


async def test_overwrite_existing_file(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test overwriting an existing file."""
    file_path = temp_work_dir / "existing.txt"
    original_content = "Original content"
    await file_path.write_text(original_content)

    new_content = "New content"
    result = await write_file_tool(Params(path=str(file_path), content=new_content))

    assert not result.is_error
    assert "successfully overwritten" in result.message
    assert await file_path.read_text() == new_content


async def test_append_to_file(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test appending to an existing file."""
    file_path = temp_work_dir / "append_test.txt"
    original_content = "First line\n"
    await file_path.write_text(original_content)

    append_content = "Second line\n"
    result = await write_file_tool(
        Params(path=str(file_path), content=append_content, mode="append")
    )

    assert not result.is_error
    assert "successfully appended to" in result.message
    expected_content = original_content + append_content
    assert await file_path.read_text() == expected_content


async def test_write_unicode_content(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing unicode content."""
    file_path = temp_work_dir / "unicode.txt"
    content = "Hello 世界 🌍\nUnicode: café, naïve, résumé"

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert await file_path.exists()
    assert await file_path.read_text(encoding="utf-8") == content


async def test_write_empty_content(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing empty content."""
    file_path = temp_work_dir / "empty.txt"
    content = ""

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert await file_path.exists()
    assert await file_path.read_text() == content


async def test_write_multiline_content(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing multiline content."""
    file_path = temp_work_dir / "multiline.txt"
    content = "Line 1\nLine 2\nLine 3\n"

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert await file_path.read_text() == content


async def test_write_with_relative_path(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing with a relative path inside the work directory."""
    relative_dir = temp_work_dir / "relative" / "path"
    await relative_dir.mkdir(parents=True, exist_ok=True)

    result = await write_file_tool(Params(path="relative/path/file.txt", content="content"))

    assert not result.is_error
    assert await (temp_work_dir / "relative" / "path" / "file.txt").read_text() == "content"


async def test_write_outside_work_directory(write_file_tool: WriteFile, outside_file: Path):
    """Test writing outside the working directory with an absolute path."""
    result = await write_file_tool(Params(path=str(outside_file), content="content"))

    assert not result.is_error
    assert outside_file.read_text() == "content"


async def test_write_outside_work_directory_with_prefix(
    write_file_tool: WriteFile, temp_work_dir: KaosPath
):
    """Paths sharing the same prefix as work dir should still be writable with absolute paths."""
    base = Path(str(temp_work_dir))
    sneaky_dir = base.parent / f"{base.name}-sneaky"
    sneaky_dir.mkdir(parents=True, exist_ok=True)
    sneaky_file = sneaky_dir / "file.txt"

    result = await write_file_tool(Params(path=str(sneaky_file), content="content"))

    assert not result.is_error
    assert sneaky_file.read_text() == "content"


async def test_write_to_nonexistent_directory(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing to a non-existent directory."""
    file_path = temp_work_dir / "nonexistent" / "file.txt"

    result = await write_file_tool(Params(path=str(file_path), content="content"))

    assert result.is_error
    assert "parent directory does not exist" in result.message


async def test_write_with_invalid_mode(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing with an invalid mode."""
    file_path = temp_work_dir / "test.txt"

    with pytest.raises(ValidationError):
        await write_file_tool(Params(path=str(file_path), content="content", mode="invalid"))  # type: ignore[reportArgumentType]


async def test_append_to_nonexistent_file(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test appending to a non-existent file (should create it)."""
    file_path = temp_work_dir / "new_append.txt"
    content = "New content\n"

    result = await write_file_tool(Params(path=str(file_path), content=content, mode="append"))

    assert not result.is_error
    assert "successfully appended to" in result.message
    assert await file_path.exists()
    assert await file_path.read_text() == content


async def test_write_large_content(write_file_tool: WriteFile, temp_work_dir: KaosPath):
    """Test writing large content."""
    file_path = temp_work_dir / "large.txt"
    content = "Large content line\n" * 1000

    result = await write_file_tool(Params(path=str(file_path), content=content))

    assert not result.is_error
    assert await file_path.exists()
    assert await file_path.read_text() == content


async def test_write_file_cannot_mutate_managed_wiki(builtin_args, tmp_path: Path) -> None:
    manager = WikiManager(tmp_path / "wiki", wal=False)
    try:
        runtime = SimpleNamespace(
            builtin_args=builtin_args,
            additional_dirs=[],
            wiki=manager,
        )
        tool = WriteFile(cast("Runtime", runtime), Approval(yolo=True))

        result = await tool(Params(path=str(manager.layout.index), content="x"))

        assert result.is_error
        assert "Wiki tool" in result.message
        assert manager.layout.index.read_text(encoding="utf-8") != "x"
    finally:
        manager.close()


async def test_write_file_rejects_symlink_alias_of_managed_wiki(
    builtin_args, tmp_path: Path
) -> None:
    manager = WikiManager(tmp_path / "wiki", wal=False)
    alias = tmp_path / "wiki-alias"
    alias.symlink_to(manager.layout.root, target_is_directory=True)
    try:
        runtime = SimpleNamespace(
            builtin_args=builtin_args,
            additional_dirs=[],
            wiki=manager,
        )
        tool = WriteFile(cast("Runtime", runtime), Approval(yolo=True))

        result = await tool(Params(path=str(alias / "index.md"), content="x"))

        assert result.is_error
        assert "Wiki tool" in result.message
    finally:
        manager.close()


async def test_write_file_rechecks_target_after_approval_symlink_swap(
    builtin_args, tmp_path: Path
) -> None:
    manager = WikiManager(tmp_path / "wiki", wal=False)
    target = Path(str(builtin_args.KIMI_WORK_DIR)) / "target.txt"
    target.write_text("safe", encoding="utf-8")

    class SwapApproval:
        async def request(self, *_args, **_kwargs):
            replacement = target.with_name("replacement-link.txt")
            replacement.symlink_to(manager.layout.index)
            os.replace(replacement, target)
            return ApprovalResult(approved=True)

    try:
        runtime = SimpleNamespace(builtin_args=builtin_args, additional_dirs=[], wiki=manager)
        tool = WriteFile(cast("Runtime", runtime), cast("Approval", SwapApproval()))

        result = await tool(Params(path=str(target), content="must not reach Wiki"))

        assert result.is_error
        assert "Wiki tool" in result.message
        assert "must not reach Wiki" not in manager.layout.index.read_text(encoding="utf-8")
    finally:
        manager.close()


async def test_write_file_rechecks_target_after_approval_hardlink_swap(
    builtin_args, tmp_path: Path
) -> None:
    manager = WikiManager(tmp_path / "wiki", wal=False)
    target = Path(str(builtin_args.KIMI_WORK_DIR)) / "target.txt"
    target.write_text("safe", encoding="utf-8")

    class SwapApproval:
        async def request(self, *_args, **_kwargs):
            replacement = target.with_name("replacement-hardlink.txt")
            os.link(manager.layout.index, replacement)
            os.replace(replacement, target)
            return ApprovalResult(approved=True)

    try:
        runtime = SimpleNamespace(builtin_args=builtin_args, additional_dirs=[], wiki=manager)
        tool = WriteFile(cast("Runtime", runtime), cast("Approval", SwapApproval()))

        result = await tool(Params(path=str(target), content="must not reach Wiki"))

        assert result.is_error
        assert "Wiki tool" in result.message
        assert "must not reach Wiki" not in manager.layout.index.read_text(encoding="utf-8")
    finally:
        manager.close()
