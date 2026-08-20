"""Two promises the skill layer used to break.

A skill turned off in the panel should stay off, and a skills directory the
user asked to be left alone should be left alone.
"""

from __future__ import annotations

import json
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

    A second built-in stays in place because the rule only holds while this
    build ships any: with none, such a name came from a catalogue that was
    removed and cannot come back, and the row answers "skill not found" to
    every operation. See TestTheListDoesNotFillWithDeadRows.
    """
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    manager = SkillManager(builtin, managed / "skill")
    _skill(builtin, "ghost")
    _skill(builtin, "still-shipped")
    manager.disable("ghost")

    (builtin / "ghost" / "SKILL.md").unlink()
    (builtin / "ghost").rmdir()

    listed = {s.name: s for s in manager.list_skills()}
    assert "ghost" in listed
    assert listed["ghost"].enabled is False


def _legacy_state(managed: Path) -> None:
    """A state file written before skills defaulted to off.

    Under that rule everything discovered is on unless named. Tests about
    discovery use it so they keep exercising discovery rather than the default.
    """
    import json

    state = SkillManager().state_file
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps({"version": 1, "disabled": [], "deleted": [], "revision": 0}),
        encoding="utf-8",
    )


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

    The state is written as a pre-default-flip one on purpose: what is being
    checked is that an unmanageable name survives discovery, not what the
    default is, and a test that stops exercising its subject when a default
    changes was not testing the subject.
    """
    user = tmp_path / "user"
    for name in ("写作助手", "my_skill", "ok-skill"):
        _skill(user, name)
    _legacy_state(managed)

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
    _legacy_state(managed)
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


class TestSkillsAreOffUntilAskedFor:
    """Every discovered skill costs a few hundred characters of context on every
    request, whether or not it is ever used, so a fresh install carries none.

    The state used to be a list of what was off, which made "on" the answer to
    every question nobody had answered — including for a skill that arrived
    yesterday from a directory the user forgot they had.
    """

    def _manager(self, tmp_path: Path, state: dict | None = None) -> SkillManager:
        manager = SkillManager()
        manager.state_file = tmp_path / "skill-state.json"
        if state is not None:
            manager.state_file.write_text(json.dumps(state), encoding="utf-8")
        return manager

    def test_a_fresh_install_carries_nothing(self, tmp_path: Path) -> None:
        assert self._manager(tmp_path).is_enabled("pdf") is False

    def test_an_existing_catalogue_is_not_emptied_by_upgrading(self, tmp_path: Path) -> None:
        """The one thing this change must not do.

        A state file written before the default flipped records what its owner
        turned *off*. Reading it under the new rule would turn everything off,
        and the first sign would be an agent that no longer knows how to do
        something it did last week.
        """
        manager = self._manager(
            tmp_path, {"version": 1, "disabled": ["docx"], "deleted": [], "revision": 3}
        )

        assert manager.is_enabled("pdf") is True
        assert manager.is_enabled("docx") is False, "and the choice they made is kept"

    def test_once_the_list_exists_it_is_the_answer(self, tmp_path: Path) -> None:
        manager = self._manager(
            tmp_path,
            {"version": 1, "disabled": [], "deleted": [], "revision": 1, "enabled": ["pdf"]},
        )

        assert manager.is_enabled("pdf") is True
        assert manager.is_enabled("docx") is False

    def test_the_switches_move_the_name_between_lists(self, tmp_path: Path) -> None:
        manager = self._manager(
            tmp_path,
            {"version": 1, "disabled": [], "deleted": [], "revision": 1, "enabled": []},
        )

        manager.enable("docx")
        assert manager.is_enabled("docx") is True

        manager.disable("docx")
        assert manager.is_enabled("docx") is False

    def test_a_corrupt_state_does_not_switch_everything_back_on(self, tmp_path: Path) -> None:
        """An unreadable file is not evidence that someone chose to carry every skill."""
        manager = self._manager(tmp_path)
        manager.state_file.write_text("{ not json", encoding="utf-8")

        assert manager.is_enabled("pdf") is False

    def test_a_name_that_cannot_be_normalised_is_off_and_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """This runs inside agent creation, before there is any UI to report to.

        One `写作助手/` in a skills directory used to end every session at startup
        with an exit code and no visible reason. The answer changed with the
        default; that it must not raise did not.
        """
        assert self._manager(tmp_path).is_enabled("写作助手") is False


class TestASkillTheRulesCannotNameIsStillAddressable:
    """`写作助手/` has no managed name, and both defaults would strand it.

    Under "on unless listed" it was permanently on and could not be switched
    off. Reversing the default without this would have made it permanently off
    and unable to be switched on, which is a deletion nobody performed and
    nothing reports.
    """

    def _manager(self, tmp_path: Path) -> SkillManager:
        manager = SkillManager()
        manager.state_file = tmp_path / "skill-state.json"
        return manager

    def test_it_starts_off_like_everything_else(self, tmp_path: Path) -> None:
        assert self._manager(tmp_path).is_enabled("写作助手") is False

    def test_it_can_be_switched_on(self, tmp_path: Path) -> None:
        manager = self._manager(tmp_path)

        manager.enable("写作助手")

        assert manager.is_enabled("写作助手") is True

    def test_and_off_again(self, tmp_path: Path) -> None:
        manager = self._manager(tmp_path)
        manager.enable("写作助手")

        manager.disable("写作助手")

        assert manager.is_enabled("写作助手") is False

    def test_switching_it_never_raises(self, tmp_path: Path) -> None:
        """This runs inside agent creation, before there is any UI to report to."""
        manager = self._manager(tmp_path)

        for name in ("写作助手", " leading space", "-leading-symbol", "with space"):
            manager.enable(name)
            assert manager.is_enabled(name) is True
            manager.disable(name)
            assert manager.is_enabled(name) is False


class TestTheListDoesNotFillWithDeadRows:
    """A disabled name with no copy is a switch or a dead row, depending.

    It is a switch when the name can still be discovered from a user or project
    directory: the disabled entry suppresses it, and hiding the row would leave
    it permanently off with nothing to turn on. That is why they are listed.

    It is a dead row when this build ships no built-in skills, because then the
    name came from a catalogue that was removed and cannot come back. Every
    operation on it answers "skill not found". Measured on the first store this
    was tried against: 298 such rows against one real skill.
    """

    def _manager(self, tmp_path: Path, builtins: bool) -> SkillManager:
        builtin_dir = tmp_path / "builtin"
        builtin_dir.mkdir()
        if builtins:
            _skill(builtin_dir, "still-shipped")
        manager = SkillManager(builtin_dir, tmp_path / "skill")
        manager.state_file = tmp_path / "skill-state.json"
        manager.state_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "disabled": ["gone-one", "gone-two", "gone-three"],
                    "deleted": [],
                    "revision": 1,
                }
            ),
            encoding="utf-8",
        )
        return manager

    def test_they_are_listed_while_built_ins_still_ship(self, tmp_path: Path) -> None:
        """The case the listing was written for: the name may live in a user dir."""
        listed = {s.name for s in self._manager(tmp_path, builtins=True).list_skills()}

        assert "gone-one" in listed
        assert "still-shipped" in listed

    def test_they_are_not_listed_when_nothing_is_bundled(self, tmp_path: Path) -> None:
        listed = [s.name for s in self._manager(tmp_path, builtins=False).list_skills()]

        assert listed == [], "rows that answer 'skill not found' to everything"

    def test_a_real_installed_skill_still_shows(self, tmp_path: Path) -> None:
        """The row that is left has to be the one that works."""
        manager = self._manager(tmp_path, builtins=False)
        manager.install_skill_md(
            "---\nname: mine\ndescription: installed by me\n---\nDo the thing.\n"
        )

        listed = [s.name for s in manager.list_skills()]

        assert listed == ["mine"]
        assert manager.get("mine") is not None
