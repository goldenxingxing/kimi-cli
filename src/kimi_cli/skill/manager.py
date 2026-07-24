"""Managed built-in and writable skill layers."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

from kimi_cli.skill import get_builtin_skills_dir
from kimi_cli.utils.frontmatter import parse_frontmatter

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def get_managed_skill_dir() -> Path:
    configured = os.environ.get("OPENKIMO_SKILL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif __import__("sys").platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "OpenKimo" / "skill"


def normalize_managed_skill_name(name: str) -> str:
    value = name.strip()
    if not _NAME_RE.fullmatch(value):
        raise ValueError("Invalid skill name")
    return value.casefold()


class SkillState(TypedDict):
    version: int
    disabled: list[str]
    deleted: list[str]
    revision: int


@dataclass(frozen=True, slots=True)
class ManagedSkill:
    name: str
    description: str
    origin: str
    enabled: bool
    deleted: bool
    modified: bool
    files: tuple[str, ...]


class SkillManager:
    def __init__(
        self,
        builtin_dir: Path | None = None,
        writable_dir: Path | None = None,
    ) -> None:
        self.builtin_dir = (builtin_dir or get_builtin_skills_dir()).resolve()
        self.writable_dir = (writable_dir or get_managed_skill_dir()).resolve()
        self.state_file = self.writable_dir.parent / "skill-state.json"
        self._recover_backups()

    @contextmanager
    def _mutation_lock(self, name: str):
        """Cross-process lock for one logical skill or the shared state."""
        lock_dir = self.writable_dir.parent / ".skill-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{normalize_managed_skill_name(name)}.lock"
        with lock_path.open("a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _replace_directory(staged: Path, destination: Path) -> None:
        """Swap a directory with rollback if the second rename fails."""
        backup = destination.with_name(f".{destination.name}.backup")
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        if backup.exists():
            shutil.rmtree(backup)
        had_destination = destination.exists()
        if had_destination:
            os.replace(destination, backup)
        try:
            os.replace(staged, destination)
        except BaseException:
            if had_destination and backup.exists():
                os.replace(backup, destination)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)

    def _recover_backups(self) -> None:
        """Restore or remove leftovers from an interrupted directory swap."""
        if not self.writable_dir.is_dir():
            return
        for backup in self.writable_dir.glob(".*.backup"):
            if not backup.is_dir():
                continue
            destination_name = backup.name[1:-len(".backup")]
            try:
                key = normalize_managed_skill_name(destination_name)
            except ValueError:
                continue
            with self._mutation_lock(key):
                destination = self.writable_dir / destination_name
                if destination.exists():
                    shutil.rmtree(backup)
                else:
                    os.replace(backup, destination)

    def _load_state(self) -> SkillState:
        if not self.state_file.is_file():
            return {"version": 1, "disabled": [], "deleted": [], "revision": 0}
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError
            raw_state = cast(dict[str, object], raw)
            disabled_raw = raw_state.get("disabled", [])
            deleted_raw = raw_state.get("deleted", [])
            revision_raw = raw_state.get("revision", 0)
            return {
                "version": 1,
                "disabled": (
                    [
                        item
                        for item in cast(list[object], disabled_raw)
                        if isinstance(item, str)
                    ]
                    if isinstance(disabled_raw, list)
                    else []
                ),
                "deleted": (
                    [
                        item
                        for item in cast(list[object], deleted_raw)
                        if isinstance(item, str)
                    ]
                    if isinstance(deleted_raw, list)
                    else []
                ),
                "revision": revision_raw if isinstance(revision_raw, int) else 0,
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            quarantine = self.state_file.with_suffix(".corrupt")
            try:
                os.replace(self.state_file, quarantine)
            except OSError:
                pass
            return {"version": 1, "disabled": [], "deleted": [], "revision": 0}

    def _save_state(self, state: SkillState) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state["revision"] = int(state.get("revision", 0)) + 1
        fd, temp_name = tempfile.mkstemp(prefix=".skill-state-", dir=self.state_file.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.state_file)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    @property
    def revision(self) -> int:
        return int(self._load_state()["revision"])

    def is_enabled(self, name: str) -> bool:
        key = normalize_managed_skill_name(name)
        state = self._load_state()
        return key not in state["disabled"] and key not in state["deleted"]

    @staticmethod
    def _directories(root: Path) -> dict[str, Path]:
        if not root.is_dir():
            return {}
        result: dict[str, Path] = {}
        for child in root.iterdir():
            if child.is_dir() and (child / "SKILL.md").is_file():
                result.setdefault(normalize_managed_skill_name(child.name), child)
        return result

    @staticmethod
    def _describe(path: Path) -> tuple[str, str]:
        content = (path / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        frontmatter = parse_frontmatter(content) or {}
        name = str(frontmatter.get("name") or path.name)
        description = str(frontmatter.get("description") or "No description provided.")
        return name, description

    @staticmethod
    def _files(path: Path) -> tuple[str, ...]:
        return tuple(
            sorted(
                str(file.relative_to(path))
                for file in path.rglob("*")
                if file.is_file() and not file.is_symlink()
            )
        )

    def list_skills(self) -> list[ManagedSkill]:
        state = self._load_state()
        disabled = set(state["disabled"])
        deleted = set(state["deleted"])
        builtins = self._directories(self.builtin_dir)
        writable = self._directories(self.writable_dir)
        result: list[ManagedSkill] = []
        for key in sorted(builtins.keys() | writable.keys()):
            builtin = builtins.get(key)
            override = writable.get(key)
            selected = override or builtin
            assert selected is not None
            name, description = self._describe(selected)
            is_deleted = key in deleted and override is None
            result.append(
                ManagedSkill(
                    name=name,
                    description=description,
                    origin="builtin" if builtin else "user",
                    enabled=key not in disabled and not is_deleted,
                    deleted=is_deleted,
                    modified=builtin is not None and override is not None,
                    files=self._files(selected),
                )
            )
        return result

    def get(self, name: str) -> ManagedSkill:
        key = normalize_managed_skill_name(name)
        for skill in self.list_skills():
            if normalize_managed_skill_name(skill.name) == key:
                return skill
        raise KeyError(name)

    def read_file(self, name: str, relative_path: str) -> str:
        key = normalize_managed_skill_name(name)
        selected = self._directories(self.writable_dir).get(key) or self._directories(
            self.builtin_dir
        ).get(key)
        if selected is None:
            raise KeyError(name)
        target = (selected / relative_path).resolve()
        if not target.is_relative_to(selected.resolve()) or not target.is_file():
            raise ValueError("Invalid skill file")
        return target.read_text(encoding="utf-8")

    def _change_set(
        self, field: Literal["disabled", "deleted"], name: str, present: bool
    ) -> None:
        key = normalize_managed_skill_name(name)
        self.get(name)
        with self._mutation_lock("state"):
            state = self._load_state()
            values: set[str] = set(state[field])
            if present:
                values.add(key)
            else:
                values.discard(key)
            state[field] = sorted(values)
            self._save_state(state)

    def disable(self, name: str) -> None:
        self._change_set("disabled", name, True)

    def enable(self, name: str) -> None:
        key = normalize_managed_skill_name(name)
        with self._mutation_lock("state"):
            state = self._load_state()
            state["disabled"] = sorted(set(state["disabled"]) - {key})
            state["deleted"] = sorted(set(state["deleted"]) - {key})
            self._save_state(state)

    def delete(self, name: str) -> None:
        key = normalize_managed_skill_name(name)
        skill = self.get(name)
        with self._mutation_lock(key):
            writable = self._directories(self.writable_dir).get(key)
            if writable:
                shutil.rmtree(writable)
        with self._mutation_lock("state"):
            state = self._load_state()
            if skill.origin == "builtin":
                state["deleted"] = sorted(set(state["deleted"]) | {key})
            state["disabled"] = sorted(set(state["disabled"]) - {key})
            self._save_state(state)

    def restore(self, name: str) -> None:
        key = normalize_managed_skill_name(name)
        if key not in self._directories(self.builtin_dir):
            raise ValueError("Only built-in skills can be restored")
        with self._mutation_lock(key):
            writable = self._directories(self.writable_dir).get(key)
            if writable:
                shutil.rmtree(writable)
        with self._mutation_lock("state"):
            state = self._load_state()
            state["disabled"] = sorted(set(state["disabled"]) - {key})
            state["deleted"] = sorted(set(state["deleted"]) - {key})
            self._save_state(state)

    def write_skill_md(self, name: str, content: str) -> ManagedSkill:
        key = normalize_managed_skill_name(name)
        frontmatter = parse_frontmatter(content) or {}
        edited_name = frontmatter.get("name")
        if edited_name is not None and (
            not isinstance(edited_name, str)
            or normalize_managed_skill_name(edited_name) != key
        ):
            raise ValueError("Editing SKILL.md cannot change the skill name")
        builtins = self._directories(self.builtin_dir)
        writable = self._directories(self.writable_dir)
        source = writable.get(key) or builtins.get(key)
        if source is None:
            raise KeyError(name)
        with self._mutation_lock(key):
            self.writable_dir.mkdir(parents=True, exist_ok=True)
            destination = self.writable_dir / source.name
            temp = Path(tempfile.mkdtemp(prefix=".skill-edit-", dir=self.writable_dir))
            try:
                shutil.copytree(source, temp / source.name)
                edited = temp / source.name
                (edited / "SKILL.md").write_text(content, encoding="utf-8")
                self._replace_directory(edited, destination)
            finally:
                shutil.rmtree(temp, ignore_errors=True)
        self.enable(name)
        return self.get(name)

    def install_archive(self, data: bytes, *, replace: bool = False) -> ManagedSkill:
        from kimi_cli.skill.archive import extract_skill_archive

        self.writable_dir.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".skill-upload-", dir=self.writable_dir))
        try:
            prepared = extract_skill_archive(data, temp)
            key = normalize_managed_skill_name(prepared.name)
            with self._mutation_lock(key):
                writable = self._directories(self.writable_dir)
                existing = {
                    **self._directories(self.builtin_dir),
                    **writable,
                }
                if key in existing and not replace:
                    raise FileExistsError(prepared.name)
                existing_path = existing.get(key)
                destination = writable.get(key) or self.writable_dir / (
                    existing_path.name if existing_path else prepared.name
                )
                staged = temp / ".staged"
                shutil.copytree(prepared.directory, staged)
                self._replace_directory(staged, destination)
        finally:
            shutil.rmtree(temp, ignore_errors=True)
        self.enable(prepared.name)
        return self.get(prepared.name)

    def install_skill_md(self, content: str, *, replace: bool = False) -> ManagedSkill:
        frontmatter = parse_frontmatter(content) or {}
        raw_name = frontmatter.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("A standalone SKILL.md must declare a name")
        name = raw_name.strip()
        key = normalize_managed_skill_name(name)
        with self._mutation_lock(key):
            self.writable_dir.mkdir(parents=True, exist_ok=True)
            writable = self._directories(self.writable_dir)
            existing = {
                **self._directories(self.builtin_dir),
                **writable,
            }
            if key in existing and not replace:
                raise FileExistsError(name)
            existing_path = existing.get(key)
            destination = writable.get(key) or self.writable_dir / (
                existing_path.name if existing_path else name
            )
            temp = Path(tempfile.mkdtemp(prefix=".skill-markdown-", dir=self.writable_dir))
            try:
                staged = temp / name
                staged.mkdir()
                (staged / "SKILL.md").write_text(content, encoding="utf-8")
                self._replace_directory(staged, destination)
            finally:
                shutil.rmtree(temp, ignore_errors=True)
        self.enable(name)
        return self.get(name)
