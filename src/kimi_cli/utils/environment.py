from __future__ import annotations

import asyncio
import ntpath
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kaos.path import KaosPath

from kimi_cli.utils.logging import logger

#: Env var pointing at any POSIX shell, honoured on every platform. The escape
#: hatch of last resort: it skips discovery entirely.
SHELL_PATH_ENV = "KIMI_CLI_SHELL_PATH"

#: Filename of the optional portable shell shipped with (or dropped next to)
#: kimi-cli on Windows. It is invoked as ``<path> -c <command>``, so a busybox
#: build has to be *copied to this name* — busybox dispatches on argv[0].
_BUNDLED_SHELL_NAME = "sh.exe"


class GitBashNotFoundError(RuntimeError):
    """Raised when kimi-cli runs on Windows but cannot locate git-bash.

    git-bash (from Git for Windows) is required because kimi-cli's Shell tool
    runs commands through bash, not PowerShell.

    Note that this is *not* fatal: :meth:`Environment.detect` catches it, falls
    back to the bundled portable shell, and otherwise starts without a shell at
    all. Only the Shell tool is lost — everything else keeps working.
    """


_GIT_BASH_INSTALL_HINT = (
    "kimi-cli on Windows runs shell commands through bash, not PowerShell, and no "
    "bash was found. Any one of these fixes it:\n"
    "  * Install Git for Windows (https://git-scm.com/downloads/win) for its bundled bash.\n"
    "  * Point KIMI_CLI_GIT_BASH_PATH at an existing bash.exe, e.g.\n"
    "    KIMI_CLI_GIT_BASH_PATH=C:\\Program Files\\Git\\bin\\bash.exe\n"
    "  * Point KIMI_CLI_SHELL_PATH at any POSIX shell to use it as-is.\n"
    f"  * Drop a portable shell at %USERPROFILE%\\.kimi\\bin\\{_BUNDLED_SHELL_NAME}."
)
_GIT_EXEC_PATH_TIMEOUT_SECONDS = 5


@dataclass(slots=True, frozen=True, kw_only=True)
class Environment:
    os_kind: Literal["Windows", "Linux", "macOS"] | str
    os_arch: str
    os_version: str
    #: None when no shell could be found. Only Windows can reach that state.
    shell_name: Literal["bash", "sh"] | None
    shell_path: KaosPath | None
    #: Why no shell was found, phrased for the user. Set iff shell_path is None.
    shell_error: str | None = None

    @property
    def has_shell(self) -> bool:
        return self.shell_path is not None

    @property
    def shell_description(self) -> str:
        """One-line shell summary for prompts and tool descriptions."""
        if self.shell_path is None:
            return "unavailable (no shell found on this machine)"
        return f"{self.shell_name} (`{self.shell_path}`)"

    @staticmethod
    async def detect() -> Environment:
        """Describe the host. Never raises — a missing shell is a value, not an error.

        Failing to find bash used to abort startup, which turned "the Shell tool
        will not work" into "the product does not open". Everything that does not
        shell out — chat, file edits, the web UI, most skills — has no reason to
        care, so the missing shell is carried as data and reported by the one
        tool that needs it.
        """
        match platform.system():
            case "Darwin":
                os_kind = "macOS"
            case "Windows":
                os_kind = "Windows"
            case "Linux":
                os_kind = "Linux"
            case system:
                os_kind = system

        os_arch = platform.machine()
        os_version = platform.version()

        shell_name: Literal["bash", "sh"] | None
        shell_path: KaosPath | None
        shell_error: str | None = None

        override = await _shell_override_path()
        if override is not None:
            shell_path = override
            shell_name = _shell_name_for(override)
        elif os_kind == "Windows":
            try:
                shell_path = await find_git_bash_path()
                shell_name = "bash"
            except GitBashNotFoundError as exc:
                # A real bash is preferred, but a portable one is far better
                # than no shell at all.
                bundled = await find_bundled_shell()
                if bundled is not None:
                    shell_path = bundled
                    shell_name = _shell_name_for(bundled)
                else:
                    shell_path = None
                    shell_name = None
                    shell_error = str(exc)
        else:
            possible_paths = [
                KaosPath("/bin/bash"),
                KaosPath("/usr/bin/bash"),
                KaosPath("/usr/local/bin/bash"),
            ]
            fallback_path = KaosPath("/bin/sh")
            for path in possible_paths:
                if await path.is_file():
                    shell_name = "bash"
                    shell_path = path
                    break
            else:
                shell_name = "sh"
                shell_path = fallback_path

        return Environment(
            os_kind=os_kind,
            os_arch=os_arch,
            os_version=os_version,
            shell_name=shell_name,
            shell_path=shell_path,
            shell_error=shell_error,
        )


def _shell_name_for(path: KaosPath) -> Literal["bash", "sh"]:
    """Classify a shell by filename. Anything not named bash is treated as sh.

    The name reaches the model through the system prompt, and calling busybox
    ash "bash" would invite bashisms it cannot run.
    """
    stem = os.path.splitext(os.path.basename(str(path)))[0].casefold()
    return "bash" if stem == "bash" else "sh"


async def _shell_override_path() -> KaosPath | None:
    """Resolve ``KIMI_CLI_SHELL_PATH``, ignoring it (with a warning) if unusable."""
    configured = os.environ.get(SHELL_PATH_ENV)
    if not configured:
        return None
    candidate = KaosPath(configured)
    if await candidate.is_file():
        return candidate
    logger.warning(
        "{env} points to {path} but no file exists there; falling back to discovery.",
        env=SHELL_PATH_ENV,
        path=configured,
    )
    return None


async def find_bundled_shell() -> KaosPath | None:
    """Locate the portable shell shipped with kimi-cli, if one is present.

    Resolution mirrors the bundled ripgrep (``share dir`` first, then the
    package's own ``deps/bin``) so an install can be topped up without being
    rebuilt. Windows-only by design: every POSIX host already has ``/bin/sh``.
    """
    if not is_windows():
        return None

    import kimi_cli
    from kimi_cli.share import get_share_dir

    package_dir = Path(kimi_cli.__file__).parent
    candidates = [
        KaosPath(str(get_share_dir() / "bin" / _BUNDLED_SHELL_NAME)),
        KaosPath(str(package_dir / "deps" / "bin" / _BUNDLED_SHELL_NAME)),
    ]
    for candidate in candidates:
        if await candidate.is_file():
            return candidate
    return None


def is_windows() -> bool:
    """Return True iff the current process is running on native Windows."""
    return platform.system() == "Windows"


async def find_git_bash_path() -> KaosPath:
    """Locate ``bash.exe`` from Git for Windows.

    Resolution order:
      1. ``KIMI_CLI_GIT_BASH_PATH`` environment variable (validated to exist).
      2. ``where.exe git`` -> ``<gitDir>/../bin/bash.exe``.
      3. ``git --exec-path`` -> Git for Windows install root -> ``bin\\bash.exe``.
      4. Common install locations (``C:\\Program Files\\Git\\bin\\bash.exe``).

    Raises:
        GitBashNotFoundError: if no candidate path resolves to an existing file.
    """
    override = os.environ.get("KIMI_CLI_GIT_BASH_PATH")
    if override:
        candidate = KaosPath(override)
        if await candidate.is_file():
            return candidate
        raise GitBashNotFoundError(
            f"KIMI_CLI_GIT_BASH_PATH points to {override} but no file exists there.\n\n"
            + _GIT_BASH_INSTALL_HINT
        )

    for git_path in await _find_git_executables():
        bash_candidate = _git_bash_candidate_from_git_path(git_path)
        if await bash_candidate.is_file():
            return bash_candidate

        git_exec_path = await asyncio.to_thread(_git_exec_path, git_path)
        if git_exec_path is None:
            continue

        for bash_candidate in _git_bash_candidates_from_exec_path(git_exec_path):
            if await bash_candidate.is_file():
                return bash_candidate

    fallback_candidates = [
        KaosPath(r"C:\Program Files\Git\bin\bash.exe"),
        KaosPath(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    ]
    for candidate in fallback_candidates:
        if await candidate.is_file():
            return candidate

    raise GitBashNotFoundError(_GIT_BASH_INSTALL_HINT)


def _git_bash_candidate_from_git_path(git_path: str) -> KaosPath:
    # git.exe usually lives at <git>/cmd/git.exe; bash.exe is at <git>/bin/bash.exe.
    # Use ntpath explicitly so this works regardless of the host OS that imports
    # this module (tests on macOS pass Windows-style paths through this code).
    return KaosPath(ntpath.join(ntpath.dirname(git_path), "..", "bin", "bash.exe"))


def _git_exec_path(git_path: str) -> str | None:
    try:
        result = subprocess.run(
            [git_path, "--exec-path"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=_GIT_EXEC_PATH_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        exec_path = line.strip()
        if exec_path:
            return exec_path
    return None


def _git_bash_candidates_from_exec_path(exec_path: str) -> list[KaosPath]:
    normalized_exec_path = ntpath.normpath(exec_path)
    install_root = _git_install_root_from_exec_path(normalized_exec_path)
    if install_root is not None:
        return [KaosPath(ntpath.join(install_root, "bin", "bash.exe"))]

    return [
        KaosPath(ntpath.normpath(ntpath.join(normalized_exec_path, "..", "..", "bin", "bash.exe")))
    ]


def _git_install_root_from_exec_path(exec_path: str) -> str | None:
    current = ntpath.normpath(exec_path)
    while True:
        parent, name = ntpath.split(current)
        if name.casefold() in {"mingw32", "mingw64"}:
            return parent
        if parent == current:
            return None
        current = parent


async def _find_git_executables() -> list[str]:
    """Find candidate git.exe paths on Windows, preserving PATH order."""
    candidates = await asyncio.to_thread(_where_git_executables)

    # Non-Windows test hosts do not have where.exe. Keep the helper directly
    # unit-testable there while the real Windows path still uses all where.exe hits.
    if not candidates:
        git_path = await asyncio.to_thread(shutil.which, "git")
        if isinstance(git_path, str):
            candidates.append(git_path)

    return _dedupe_paths(candidates)


def _where_git_executables() -> list[str]:
    # errors="replace": these decode child output with the locale encoding, and
    # a Windows console tool answers in the ANSI codepage — `where.exe git`
    # prints a localized "not found" line when git is absent. A strict decode
    # raises inside subprocess's own reader thread, where this function cannot
    # catch it: detection then returns nothing and the caller reports git-bash
    # as missing, which aborts worker startup. A garbled path is recoverable;
    # a dead detector is not.
    try:
        result = subprocess.run(
            ["where.exe", "git"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError:
        return []

    if result.returncode != 0:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for path in paths:
        key = path.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
