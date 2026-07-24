from pathlib import Path
import os
import pytest

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


def test_enabled_lookup_does_not_scan_all_skill_files(
    monkeypatch, tmp_path: Path
) -> None:
    manager = SkillManager(tmp_path / "builtin", tmp_path / "skill")
    monkeypatch.setattr(
        manager,
        "list_skills",
        lambda: (_ for _ in ()).throw(AssertionError("full scan")),
    )

    assert manager.is_enabled("any-valid-name") is True


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


def test_edit_rejects_logical_rename_without_changing_skill(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    writable = tmp_path / "skill"
    builtin.mkdir()
    _skill(builtin, "factory")
    manager = SkillManager(builtin, writable)

    try:
        manager.write_skill_md(
            "factory",
            "---\nname: renamed\ndescription: Renamed\n---\n",
        )
    except ValueError as error:
        assert "name" in str(error)
    else:
        raise AssertionError("logical rename should be rejected")

    assert not (writable / "factory").exists()
    assert manager.get("factory").description == "description"


def test_failed_replacement_restores_previous_skill(
    monkeypatch, tmp_path: Path
) -> None:
    builtin = tmp_path / "builtin"
    writable = tmp_path / "skill"
    builtin.mkdir()
    writable.mkdir()
    _skill(writable, "custom", "old")
    manager = SkillManager(builtin, writable)
    real_replace = os.replace
    calls = 0

    def fail_staged_swap(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr("kimi_cli.skill.manager.os.replace", fail_staged_swap)

    with pytest.raises(OSError, match="simulated"):
        manager.write_skill_md(
            "custom",
            "---\nname: custom\ndescription: new\n---\n",
        )

    assert manager.get("custom").description == "old"


def test_manager_recovers_orphaned_backup_on_startup(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    writable = tmp_path / "skill"
    builtin.mkdir()
    writable.mkdir()
    backup = _skill(writable, ".custom.backup", "old")
    (backup / "SKILL.md").write_text(
        "---\nname: custom\ndescription: old\n---\n",
        encoding="utf-8",
    )

    manager = SkillManager(builtin, writable)

    assert not backup.exists()
    assert manager.get("custom").description == "old"


@pytest.mark.parametrize("installer", ["markdown", "archive"])
def test_replacement_reuses_existing_directory_casing(
    installer: str, tmp_path: Path
) -> None:
    builtin = tmp_path / "builtin"
    writable = tmp_path / "skill"
    builtin.mkdir()
    writable.mkdir()
    _skill(writable, "Demo", "old")
    manager = SkillManager(builtin, writable)
    content = "---\nname: demo\ndescription: new\n---\n"

    if installer == "markdown":
        manager.install_skill_md(content, replace=True)
    else:
        import io
        import zipfile

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("demo/SKILL.md", content)
        manager.install_archive(archive.getvalue(), replace=True)

    assert (writable / "Demo").is_dir()
    assert sorted(path.name for path in writable.iterdir() if not path.name.startswith(".")) == [
        "Demo"
    ]
    assert manager.get("demo").description == "new"
