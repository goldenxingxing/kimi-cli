"""Open local apps for a path on the host machine."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from kimi_cli import logger

router = APIRouter(prefix="/api/open-in", tags=["open-in"])


class OpenInRequest(BaseModel):
    """Open path in a local app."""

    app: Literal["finder", "cursor", "vscode", "iterm", "terminal", "antigravity"]
    path: str


class OpenInResponse(BaseModel):
    """Open path response."""

    ok: bool
    detail: str | None = None


def _resolve_path(path: str) -> Path:
    """Resolve and validate a path (file or directory)."""
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path does not exist: {path}",
        ) from None

    if not resolved.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path does not exist: {path}",
        )
    return resolved


def _run_command(args: list[str]) -> None:
    subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )


def _spawn_process(args: list[str]) -> None:
    subprocess.Popen(args, close_fds=True)


def _open_app(app_name: str, path: Path, fallback: str | None = None) -> None:
    try:
        _run_command(["open", "-a", app_name, str(path)])
        return
    except subprocess.CalledProcessError as exc:
        if fallback is None:
            raise
        logger.warning("Open with {} failed: {}", app_name, exc)
    _run_command(["open", "-a", fallback, str(path)])


def _applescript_string(value: str) -> str:
    """Render `value` as an AppleScript string literal.

    The path is interpolated into AppleScript *source*, so it has to be quoted
    for AppleScript before `quoted form of` quotes it for the shell. A double
    quote is a legal character in a macOS filename, and without this a
    directory named `x" & (do shell script "…") & "` closed the literal and ran
    the rest of its own name — a POST body reaching osascript as code.
    """
    escaped = (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _open_terminal(path: Path) -> None:
    literal = _applescript_string(str(path))
    script = f'tell application "Terminal" to do script "cd " & quoted form of {literal}'
    _run_command(["osascript", "-e", script])


def _open_iterm(path: Path) -> None:
    script = "\n".join(
        [
            'tell application "iTerm"',
            "  create window with default profile",
            "  tell current session of current window",
            f'    write text "cd " & quoted form of {_applescript_string(str(path))}',
            "  end tell",
            "end tell",
        ]
    )
    try:
        _run_command(["osascript", "-e", script])
    except subprocess.CalledProcessError:
        script = script.replace('"iTerm"', '"iTerm2"')
        _run_command(["osascript", "-e", script])


#: Characters cmd.exe re-parses out of a command line it is handed. `%` is
#: here because %VAR% is expanded at cmd's initial parse whatever the quoting,
#: and `!` because it is, too, on a host with DelayedExpansion enabled.
_CMD_METACHARACTERS = '&|^<>"%!'


def _reject_cmd_metacharacters(path: Path) -> None:
    """Refuse a path cmd.exe would read as syntax rather than as a path.

    These helpers hand their argument to `cmd /c start ...`, and cmd re-parses
    what it receives: a directory named `x & calc` runs calc. The path comes
    from a POST body, and the agent is free to create such a directory, so
    this is the same hole the AppleScript quoting closed on macOS. Refusing is
    the honest fix — cmd's quoting rules do not survive being nested inside
    `start`, and there is no path worth opening that needs these characters.
    """
    if any(ch in str(path) for ch in _CMD_METACHARACTERS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Path contains a character the Windows shell would interpret "
                f"({_CMD_METACHARACTERS}); cannot open it."
            ),
        )


def _open_windows_app(command: str, path: Path) -> None:
    _reject_cmd_metacharacters(path)
    _run_command(["cmd", "/c", "start", "", command, str(path)])


def _open_windows_explorer(path: Path, *, is_file: bool) -> None:
    if is_file:
        _spawn_process(["explorer", f"/select,{path}"])
    else:
        _spawn_process(["explorer", str(path)])


def _open_windows_terminal(path: Path) -> None:
    _reject_cmd_metacharacters(path)
    try:
        _run_command(["cmd", "/c", "start", "", "wt.exe", "-d", str(path)])
    except subprocess.CalledProcessError as exc:
        logger.warning("Open with Windows Terminal failed: {}", exc)
        _run_command(["cmd", "/c", "start", "", "cmd.exe", "/K", f'cd /d "{path}"'])


def _open_in_macos(app: OpenInRequest, path: Path, *, is_file: bool) -> None:
    match app.app:
        case "finder":
            if is_file:
                # Reveal file in Finder
                _run_command(["open", "-R", str(path)])
            else:
                _run_command(["open", str(path)])
        case "cursor":
            _open_app("Cursor", path)
        case "vscode":
            _open_app("Visual Studio Code", path, fallback="Code")
        case "antigravity":
            _open_app("Antigravity", path)
        case "iterm":
            # Terminal apps need directory
            directory = path.parent if is_file else path
            _open_iterm(directory)
        case "terminal":
            directory = path.parent if is_file else path
            _open_terminal(directory)
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported app: {app.app}",
            )


def _open_in_windows(app: OpenInRequest, path: Path, *, is_file: bool) -> None:
    match app.app:
        case "finder":
            _open_windows_explorer(path, is_file=is_file)
        case "cursor":
            _open_windows_app("cursor", path)
        case "vscode":
            _open_windows_app("code", path)
        case "terminal":
            directory = path.parent if is_file else path
            _open_windows_terminal(directory)
        case "iterm" | "antigravity":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{app.app} is not supported on Windows.",
            )
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported app: {app.app}",
            )


def _open_in_sync(request: OpenInRequest, path: Path, *, is_file: bool) -> None:
    if sys.platform == "darwin":
        _open_in_macos(request, path, is_file=is_file)
    else:
        _open_in_windows(request, path, is_file=is_file)


@router.post("", summary="Open a path in a local application")
async def open_in(request: OpenInRequest) -> OpenInResponse:
    if sys.platform not in {"darwin", "win32"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Open-in is only supported on macOS and Windows.",
        )

    path = _resolve_path(request.path)
    is_file = path.is_file()

    try:
        await asyncio.to_thread(_open_in_sync, request, path, is_file=is_file)
    except subprocess.CalledProcessError as exc:
        logger.warning("Open-in failed ({}): {}", request.app, exc)
        detail = exc.stderr.strip() if exc.stderr else "Failed to open application."
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from exc
    except OSError as exc:
        # `open`, `osascript`, `explorer`, `wt.exe` — any of them can be
        # missing or refuse to spawn, which is not a CalledProcessError. That
        # went out as a bare 500 with a stack trace instead of a sentence
        # saying which app could not be launched.
        logger.warning("Open-in could not launch ({}): {}", request.app, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not launch {request.app}: {exc}",
        ) from exc

    return OpenInResponse(ok=True)
