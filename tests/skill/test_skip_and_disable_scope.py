"""Two promises the skill layer used to break.

A skill turned off in the panel should stay off, and a skills directory the
user asked to be left alone should be left alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kaos.path import KaosPath

from kimi_cli.skill import (
    ScopedSkillsRoot,
    discover_skills_from_roots,
    resolve_skills_roots,
)
from kimi_cli.skill.manager import SkillManager


def _skill(root: Path, name: str, description: str = "d") -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nbody",
        encoding="utf-8",
    )


@pytest.fixture
def managed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the managed-skill state at a scratch directory."""
    state_dir = tmp_path / "managed"
    state_dir.mkdir()
    monkeypatch.setenv("OPENKIMO_SKILL_DIR", str(state_dir / "skill"))
    return state_dir


async def test_disabling_a_skill_also_suppresses_a_higher_priority_copy(
    tmp_path: Path, managed: Path
) -> None:
    """The regression this file exists for.

    A same-named skill in a higher-priority root wins discovery, and the
    enabled-check used to run only against skills under the managed roots. So
    turning one off did nothing at all whenever the user kept their own copy —
    precisely the person most likely to be managing skills, being told the
    switch had worked.
    """
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    _skill(builtin, "docx", "the bundled one")
    _skill(user, "docx", "the user's own copy")

    manager = SkillManager(builtin, managed / "skill")
    manager.disable("docx")

    roots = [
        ScopedSkillsRoot(root=KaosPath(str(user)), scope="user"),
        ScopedSkillsRoot(root=KaosPath(str(builtin)), scope="builtin"),
    ]
    names = [s.name for s in await discover_skills_from_roots(roots)]

    assert names == [], "a disabled name is disabled wherever the winning copy came from"


async def test_enabling_brings_it_back(tmp_path: Path, managed: Path) -> None:
    builtin = tmp_path / "builtin"
    _skill(builtin, "docx")
    manager = SkillManager(builtin, managed / "skill")
    roots = [ScopedSkillsRoot(root=KaosPath(str(builtin)), scope="builtin")]

    manager.disable("docx")
    assert [s.name for s in await discover_skills_from_roots(roots)] == []

    manager.enable("docx")
    assert [s.name for s in await discover_skills_from_roots(roots)] == ["docx"]


def test_a_disabled_name_with_no_managed_copy_is_still_listed(
    tmp_path: Path, managed: Path
) -> None:
    """Otherwise it is off for good, with nothing in the panel to switch.

    Happens when an upgrade drops a built-in the user had disabled, or when the
    name only ever existed in a directory this manager does not own.
    """
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    manager = SkillManager(builtin, managed / "skill")
    _skill(builtin, "ghost")
    manager.disable("ghost")

    (builtin / "ghost" / "SKILL.md").unlink()
    (builtin / "ghost").rmdir()

    listed = {s.name: s for s in manager.list_skills()}
    assert "ghost" in listed
    assert listed["ghost"].enabled is False


async def test_skip_skill_dirs_drops_an_auto_discovered_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "project"
    (work / ".kimi" / "skills").mkdir(parents=True)
    monkeypatch.setattr("kimi_cli.skill.find_user_skills_dirs", _no_user_dirs)

    with_it = await resolve_skills_roots(KaosPath(str(work)))
    without = await resolve_skills_roots(
        KaosPath(str(work)), skip_skill_dirs=[str(work / ".kimi" / "skills")]
    )

    assert any(r.scope == "project" for r in with_it)
    assert not any(r.scope == "project" for r in without)


async def test_an_explicitly_requested_dir_is_never_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for a directory and skipping it is a contradiction."""
    asked = tmp_path / "asked"
    asked.mkdir()
    monkeypatch.setattr("kimi_cli.skill.find_user_skills_dirs", _no_user_dirs)

    roots = await resolve_skills_roots(
        KaosPath(str(tmp_path / "work")),
        skills_dirs=[KaosPath(str(asked))],
        skip_skill_dirs=[str(asked)],
    )

    assert any(str(r.root) == str(asked) for r in roots)


async def test_skip_entries_that_match_nothing_are_harmless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kimi_cli.skill.find_user_skills_dirs", _no_user_dirs)

    roots = await resolve_skills_roots(
        KaosPath(str(tmp_path)),
        skip_skill_dirs=["", "   ", "~/nowhere/at/all", "/does/not/exist"],
    )

    assert isinstance(roots, list)


async def _no_user_dirs(*, merge_brands: bool = False) -> list[KaosPath]:
    return []
