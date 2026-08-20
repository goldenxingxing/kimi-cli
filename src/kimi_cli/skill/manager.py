"""Managed built-in and writable skill layers."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from kimi_cli import logger
from kimi_cli.skill import get_builtin_skills_dir, normalize_skill_name
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
    enabled: NotRequired[list[str]]
    """Which skills are on, when the answer is not "all of them".

    Absent means a state file written before skills defaulted to off, and is
    read as the old behaviour — everything discovered is on unless disabled.
    Upgrading must not silently empty a catalogue someone curated.

    Present, including empty, means the list is the answer. A fresh install
    starts there: every discovered skill costs a few hundred characters of
    context on every request whether or not it is ever used, so the default is
    to carry none of them and let someone choose.
    """


@dataclass(frozen=True, slots=True)
class ManagedSkill:
    name: str
    description: str
    origin: str
    enabled: bool
    deleted: bool
    modified: bool
    files: tuple[str, ...]
    #: Category declared in SKILL.md frontmatter, verbatim. ``None`` means the
    #: skill declared none and the UI is free to classify it however it likes.
    category: str | None = None


@dataclass(frozen=True, slots=True)
class BulkSkillResult:
    """Outcome of :meth:`SkillManager.bulk_action`.

    ``missing`` holds the names that resolved to no installed skill. A bulk
    sweep reports them instead of failing: the caller is acting on a whole
    group, and one stale name must not abandon the rest.
    """

    applied: tuple[str, ...]
    missing: tuple[str, ...]


def state_key(name: str) -> str:
    """The name under which a skill's on/off state is recorded.

    The managed-skill rules reject names discovery accepts — a non-ASCII
    letter, a space, a leading symbol — so a folder called `写作助手/` has no
    managed name at all. It still has to be addressable: while the default was
    "on unless listed", such a skill was permanently on and could not be turned
    off; with the default reversed it would be permanently off and could not be
    turned on, which is a silent deletion.

    So the state falls back to the same normalization discovery uses. Both
    switches and the check go through here, which is what makes the panel able
    to address a skill it cannot otherwise manage.
    """
    try:
        return normalize_managed_skill_name(name)
    except ValueError:
        return normalize_skill_name(name)


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
            destination_name = backup.name[1 : -len(".backup")]
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
            # No file at all is a fresh install, and a fresh install carries
            # nothing until asked. An existing file without "enabled" is a
            # different thing entirely and is handled below.
            return {"version": 1, "disabled": [], "deleted": [], "revision": 0, "enabled": []}
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError
            raw_state = cast(dict[str, object], raw)
            disabled_raw = raw_state.get("disabled", [])
            deleted_raw = raw_state.get("deleted", [])
            revision_raw = raw_state.get("revision", 0)
            enabled_raw = raw_state.get("enabled")
            enabled: list[str] | None = (
                [item for item in cast(list[object], enabled_raw) if isinstance(item, str)]
                if isinstance(enabled_raw, list)
                else None
            )
            state: SkillState = {
                "version": 1,
                "disabled": (
                    [item for item in cast(list[object], disabled_raw) if isinstance(item, str)]
                    if isinstance(disabled_raw, list)
                    else []
                ),
                "deleted": (
                    [item for item in cast(list[object], deleted_raw) if isinstance(item, str)]
                    if isinstance(deleted_raw, list)
                    else []
                ),
                "revision": revision_raw if isinstance(revision_raw, int) else 0,
            }
            if enabled is not None:
                state["enabled"] = enabled
            return state
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            quarantine = self.state_file.with_suffix(".corrupt")
            with contextlib.suppress(OSError):
                os.replace(self.state_file, quarantine)
            # Fresh-install semantics, not legacy. An unreadable file is not
            # evidence that someone chose to carry every skill; reading it that
            # way would mean a corrupt state turns the whole catalogue back on.
            return {"version": 1, "disabled": [], "deleted": [], "revision": 0, "enabled": []}

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
        """Whether *name* is switched on.

        Called once per discovered skill, with whatever name its SKILL.md
        declares — and discovery normalizes with a bare casefold, so it accepts
        names these managed-name rules reject: a non-ASCII letter, a space, a
        leading symbol. Such a name cannot appear on any of these lists, since
        they hold names this same validator produced.

        A name that cannot be normalized is answered "off" rather than "on".
        That is the reverse of what it used to be, and follows the default: a
        skill nobody could switch on is not one to carry into every request.
        What has not changed is that it must not raise — this runs inside agent
        creation, in the session worker, before there is any UI to report to,
        and one `写作助手/` in a skills directory used to end every session at
        startup with an exit code and no visible reason.
        """
        state = self._load_state()
        key = state_key(name)
        if key in state["deleted"] or key in state["disabled"]:
            return False
        enabled = state.get("enabled")
        if enabled is None:
            # A state file from before skills defaulted to off. Its owner
            # curated a catalogue under the old rule; upgrading is not the
            # moment to empty it.
            return True
        return key in enabled

    @staticmethod
    def _directories(root: Path) -> dict[str, Path]:
        """Managed skill directories under *root*, keyed by normalized name.

        The writable root is a folder in Application Support that people open
        and drop skills into by hand, so a directory name the managed-name
        rules reject is a thing that happens, not a thing that cannot. Skip it:
        the panel has no way to act on a name it cannot address, and raising
        here would fail the whole listing — every manageable skill made
        unreachable by one folder beside them. The agent still discovers and
        loads it; only management passes it over.
        """
        if not root.is_dir():
            return {}
        result: dict[str, Path] = {}
        for child in root.iterdir():
            if child.is_dir() and (child / "SKILL.md").is_file():
                try:
                    key = normalize_managed_skill_name(child.name)
                except ValueError:
                    logger.warning(
                        "Skill directory {path} cannot be managed: its name is"
                        " not a valid managed skill name. It is still"
                        " discovered and loaded, but the skills panel cannot"
                        " show or toggle it.",
                        path=child,
                    )
                    continue
                result.setdefault(key, child)
        return result

    @staticmethod
    def _declared_category(frontmatter: dict[str, object]) -> str | None:
        """Read ``category`` from the frontmatter, top level or under ``metadata``.

        Both spellings are in the wild — the bundled skills that declare one all
        nest it under ``metadata`` — and neither is worth privileging.
        """
        value = frontmatter.get("category")
        if value is None:
            metadata = frontmatter.get("metadata")
            if isinstance(metadata, dict):
                value = cast(dict[str, object], metadata).get("category")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _describe(path: Path) -> tuple[str, str, str | None]:
        content = (path / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        frontmatter = parse_frontmatter(content) or {}
        name = str(frontmatter.get("name") or path.name)
        description = str(frontmatter.get("description") or "No description provided.")
        return name, description, SkillManager._declared_category(frontmatter)

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
        listed: set[str] = set()
        for key in sorted(builtins.keys() | writable.keys()):
            listed.add(key)
            builtin = builtins.get(key)
            override = writable.get(key)
            selected = override or builtin
            assert selected is not None
            name, description, category = self._describe(selected)
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
                    category=category,
                )
            )

        # A disabled name with no managed copy left — the name may still exist
        # in a user or project skills directory, where a disabled entry keeps
        # suppressing it. Leaving those off the list would make them permanently
        # off with nothing to switch, so they are shown.
        #
        # Not when this build ships no built-in skills, though. Then every such
        # name came from a catalogue that was removed and cannot come back, and
        # the panel fills with rows that answer "skill not found" to everything
        # — 298 of them on the first store this was tried against. A name that
        # refers to nothing is not a switch, it is a dead row.
        #
        # Keyed by directory name, collected above: a skill's frontmatter name
        # is free-form and need not be a valid managed-skill key, so
        # re-normalising it here would raise on perfectly good skills.
        if not builtins:
            return result

        for key in sorted(disabled - listed):
            result.append(
                ManagedSkill(
                    name=key,
                    description=(
                        "Disabled. No managed copy of this skill is installed — it may live "
                        "in a user or project skills directory, or have been removed by an "
                        "upgrade. Enable to stop suppressing it."
                    ),
                    origin="user",
                    enabled=False,
                    deleted=False,
                    modified=False,
                    files=(),
                    category=None,
                )
            )
        return result

    def name_index(self) -> dict[str, str]:
        """Map normalized *directory* name to the *displayed* skill name.

        These differ whenever a SKILL.md declares a ``name`` in its frontmatter:
        ``_directories`` keys by directory while ``_describe`` reports the
        frontmatter name. Anything that joins a filesystem path against
        :meth:`list_skills` output — such as usage analytics reconstructed from
        session logs — has to go through this alias table or the two will
        silently fail to line up.

        The writable layer wins over the builtin layer, matching
        :meth:`list_skills`.
        """
        builtins = self._directories(self.builtin_dir)
        writable = self._directories(self.writable_dir)
        index: dict[str, str] = {}
        for key in builtins.keys() | writable.keys():
            selected = writable.get(key) or builtins.get(key)
            if selected is None:  # pragma: no cover - defensive
                continue
            try:
                name, _, _ = self._describe(selected)
            except OSError:
                name = selected.name
            index[key] = name
        return index

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

    def _change_set(self, field: Literal["disabled", "deleted"], name: str, present: bool) -> None:
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
        key = state_key(name)
        with self._mutation_lock("state"):
            state = self._load_state()
            state["disabled"] = sorted(set(state["disabled"]) | {key})
            enabled = state.get("enabled")
            if enabled is not None:
                state["enabled"] = sorted(set(enabled) - {key})
            self._save_state(state)

    def enable(self, name: str) -> None:
        key = state_key(name)
        with self._mutation_lock("state"):
            state = self._load_state()
            state["disabled"] = sorted(set(state["disabled"]) - {key})
            state["deleted"] = sorted(set(state["deleted"]) - {key})
            enabled = state.get("enabled")
            if enabled is not None:
                state["enabled"] = sorted(set(enabled) | {key})
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

    def bulk_action(
        self,
        names: Iterable[str],
        action: Literal["enable", "disable", "delete"],
    ) -> BulkSkillResult:
        """Apply one action to many skills under a single state write.

        The per-skill routes would work in a loop, but each one is its own
        read-modify-write of ``skill-state.json``; sweeping a category of forty
        skills that way leaves every intermediate state observable — and a
        failure halfway through leaves half of it applied. Here the whole batch
        lands in one revision.
        """
        installed = {
            normalize_managed_skill_name(skill.name): skill for skill in self.list_skills()
        }
        keys: list[str] = []
        missing: list[str] = []
        for name in names:
            try:
                key = normalize_managed_skill_name(name)
            except ValueError:
                missing.append(name)
                continue
            if key not in installed:
                missing.append(name)
            elif key not in keys:
                keys.append(key)

        if action == "delete":
            # Drop the writable overrides first: the state write below is what
            # makes the deletion visible, so doing it last keeps a crash in the
            # middle from reporting skills as present when their files are gone.
            for key in keys:
                with self._mutation_lock(key):
                    writable = self._directories(self.writable_dir).get(key)
                    if writable:
                        shutil.rmtree(writable)

        with self._mutation_lock("state"):
            state = self._load_state()
            disabled = set(state["disabled"])
            deleted = set(state["deleted"])
            batch = set(keys)
            if action == "disable":
                disabled |= batch
            elif action == "enable":
                disabled -= batch
                deleted -= batch
            else:
                # Only built-ins are tombstoned; a user-installed skill is gone
                # with its directory. Mirrors `delete`.
                builtins = set(self._directories(self.builtin_dir))
                deleted |= batch & builtins
                disabled -= batch
            state["disabled"] = sorted(disabled)
            state["deleted"] = sorted(deleted)
            self._save_state(state)
        return BulkSkillResult(applied=tuple(keys), missing=tuple(missing))

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
            not isinstance(edited_name, str) or normalize_managed_skill_name(edited_name) != key
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
