"""Validation and bounded extraction for uploaded skill archives."""

from __future__ import annotations

import io
import shutil
import stat
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kimi_cli.skill.manager import normalize_managed_skill_name
from kimi_cli.utils.frontmatter import parse_frontmatter


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    max_entries: int = 512
    max_archive_bytes: int = 20 * 1024 * 1024
    max_expanded_bytes: int = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PreparedSkill:
    name: str
    directory: Path


def _safe_parts(raw_name: str) -> tuple[str, ...]:
    if "\\" in raw_name:
        raise ValueError("unsafe archive path")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe archive path")
    return path.parts


def extract_skill_archive(
    data: bytes,
    destination: Path,
    limits: ArchiveLimits | None = None,
) -> PreparedSkill:
    limits = limits or ArchiveLimits()
    if len(data) > limits.max_archive_bytes:
        raise ValueError("archive exceeds size limit")
    destination.mkdir(parents=True, exist_ok=True)

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid ZIP archive") from exc

    with archive:
        entries = archive.infolist()
        if not entries or len(entries) > limits.max_entries:
            raise ValueError("archive entry count exceeds limit")
        expanded = 0
        safe_entries: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
        normalized_targets: set[str] = set()
        for entry in entries:
            parts = _safe_parts(entry.filename)
            normalized_target = "/".join(
                unicodedata.normalize("NFC", part).casefold().rstrip(" .")
                for part in parts
            )
            if not normalized_target or normalized_target in normalized_targets:
                raise ValueError("archive contains duplicate normalized paths")
            normalized_targets.add(normalized_target)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("archive symbolic links are not allowed")
            expanded += entry.file_size
            if expanded > limits.max_expanded_bytes:
                raise ValueError("archive expanded size exceeds limit")
            safe_entries.append((entry, parts))

        for entry, parts in safe_entries:
            target = destination.joinpath(*parts)
            resolved = target.resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError("unsafe archive path")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)

    roots = {parts[0] for _, parts in safe_entries}
    if len(roots) == 1 and not (destination / "SKILL.md").is_file():
        skill_dir = destination / next(iter(roots))
    else:
        skill_dir = destination
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise ValueError("archive must contain exactly one skill with SKILL.md")

    content = skill_file.read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(content) or {}
    name = str(frontmatter.get("name") or skill_dir.name)
    normalize_managed_skill_name(name)
    return PreparedSkill(name=name, directory=skill_dir)
