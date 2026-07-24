import io
import zipfile
from pathlib import Path

import pytest

from kimi_cli.skill.archive import ArchiveLimits, extract_skill_archive


def _zip(files: dict[str, str]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def test_extracts_one_skill_directory(tmp_path: Path) -> None:
    data = _zip(
        {
            "demo/SKILL.md": "---\nname: demo\ndescription: Demo\n---\n",
            "demo/scripts/run.py": "print('ok')\n",
        }
    )

    prepared = extract_skill_archive(data, tmp_path)

    assert prepared.name == "demo"
    assert (prepared.directory / "SKILL.md").is_file()
    assert (prepared.directory / "scripts/run.py").is_file()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "demo/../../escape", "demo\\..\\escape"])
def test_rejects_unsafe_archive_paths(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        extract_skill_archive(_zip({name: "bad"}), tmp_path)


def test_enforces_expanded_size_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="size"):
        extract_skill_archive(
            _zip({"demo/SKILL.md": "x" * 100}),
            tmp_path,
            ArchiveLimits(max_expanded_bytes=32),
        )


@pytest.mark.parametrize(
    "files",
    [
        [
            ("demo/SKILL.md", "---\nname: demo\n---\n"),
            ("demo/SKILL.md", "---\nname: other\n---\n"),
        ],
        [
            ("demo/SKILL.md", "---\nname: demo\n---\n"),
            ("demo/skill.md", "collision"),
        ],
    ],
)
def test_rejects_normalized_target_collisions(
    tmp_path: Path, files: list[tuple[str, str]]
) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files:
            archive.writestr(name, content)

    with pytest.raises(ValueError, match="duplicate"):
        extract_skill_archive(stream.getvalue(), tmp_path)
