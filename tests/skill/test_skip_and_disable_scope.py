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


async def test_a_name_the_manager_cannot_normalize_does_not_kill_discovery(
    tmp_path: Path, managed: Path
) -> None:
    """Discovery accepts any name; the manager's rules accept far fewer.

    `discover_skills_from_roots` asks `is_enabled` about every skill it found,
    and that used to normalize the name under the managed-skill rules — which
    reject a non-ASCII letter by raising. One such folder in any skills
    directory meant `KimiCLI.create` raised, so the session worker died at
    startup, every time, with an exit code and nothing on its stderr.

    `my_skill` is here to pin the other side of the line: an underscore is
    inside the managed rules, and widening them is not what this fix does.
    """
    user = tmp_path / "user"
    for name in ("写作助手", "my_skill", "ok-skill"):
        _skill(user, name)

    roots = [ScopedSkillsRoot(root=KaosPath(str(user)), scope="user")]
    names = sorted(s.name for s in await discover_skills_from_roots(roots))

    assert names == ["my_skill", "ok-skill", "写作助手"]


async def test_unmanageable_names_survive_while_a_disabled_one_is_still_dropped(
    tmp_path: Path, managed: Path
) -> None:
    """Tolerating the names the manager cannot address must not tolerate the
    ones it can: a disabled skill stays disabled."""
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    _skill(builtin, "docx")
    _skill(user, "写作助手")
    _skill(user, "docx")
    SkillManager(builtin, managed / "skill").disable("docx")

    roots = [ScopedSkillsRoot(root=KaosPath(str(user)), scope="user")]
    names = [s.name for s in await discover_skills_from_roots(roots)]

    assert names == ["写作助手"]


def test_listing_skips_a_directory_it_cannot_manage(tmp_path: Path, managed: Path) -> None:
    """The writable root is a folder people drop skills into by hand, so a name
    outside the managed rules lands there. Listing must pass over it rather than
    fail, which would take every manageable skill beside it out of the panel."""
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    writable = managed / "skill"
    _skill(writable, "写作助手")
    _skill(writable, "usable")

    listed = [s.name for s in SkillManager(builtin, writable).list_skills()]

    assert listed == ["usable"]


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
