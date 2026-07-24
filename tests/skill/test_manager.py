from pathlib import Path

from kimi_cli.skill.manager import SkillManager


def _skill(root: Path, name: str, description: str = "description") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n",
        encoding="utf-8",
    )
    return directory


def test_manager_merges_builtin_and_writable_layers(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    writable = tmp_path / "skill"
    builtin.mkdir()
    writable.mkdir()
    _skill(builtin, "factory", "factory original")
    _skill(writable, "custom", "custom skill")
    _skill(writable, "factory", "user override")

    manager = SkillManager(builtin, writable)

    skills = {item.name: item for item in manager.list_skills()}
    assert skills["factory"].description == "user override"
    assert skills["factory"].origin == "builtin"
    assert skills["factory"].modified is True
    assert skills["custom"].origin == "user"


def test_builtin_management_is_state_only_until_edit(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    writable = tmp_path / "skill"
    builtin.mkdir()
    _skill(builtin, "factory")
    manager = SkillManager(builtin, writable)

    assert not writable.exists()
    manager.disable("factory")
    assert manager.get("factory").enabled is False
    assert not (writable / "factory").exists()

    manager.enable("factory")
    manager.write_skill_md("factory", "---\nname: factory\ndescription: edited\n---\n")
    assert (writable / "factory" / "SKILL.md").is_file()
    assert manager.get("factory").modified is True

    manager.restore("factory")
    assert not (writable / "factory").exists()
    assert manager.get("factory").description == "description"


def test_deleted_builtin_can_be_restored(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    writable = tmp_path / "skill"
    builtin.mkdir()
    _skill(builtin, "factory")
    manager = SkillManager(builtin, writable)

    manager.delete("factory")
    assert manager.get("factory").deleted is True
    assert manager.get("factory").enabled is False

    manager.restore("factory")
    assert manager.get("factory").deleted is False
    assert manager.get("factory").enabled is True
