from __future__ import annotations

import shutil
import subprocess

import pytest
from fastapi import HTTPException

from kimi_cli.web.api import open_in as open_in_api


@pytest.mark.anyio
async def test_open_in_supports_windows_directory(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(open_in_api.sys, "platform", "win32")
    monkeypatch.setattr(open_in_api, "_spawn_process", lambda args: calls.append(args))

    response = await open_in_api.open_in(
        open_in_api.OpenInRequest(app="finder", path=str(tmp_path))
    )

    assert response.ok is True
    assert calls == [["explorer", str(tmp_path)]]


@pytest.mark.anyio
async def test_open_in_supports_windows_file_selection(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(open_in_api.sys, "platform", "win32")
    monkeypatch.setattr(open_in_api, "_spawn_process", lambda args: calls.append(args))

    response = await open_in_api.open_in(
        open_in_api.OpenInRequest(app="finder", path=str(file_path))
    )

    assert response.ok is True
    assert calls == [["explorer", f"/select,{file_path}"]]


@pytest.mark.anyio
async def test_open_in_offloads_sync_work_to_thread(monkeypatch, tmp_path) -> None:
    offloaded: dict[str, object] = {}

    def fake_open_in_sync(request, path, *, is_file: bool) -> None:
        offloaded["request"] = request
        offloaded["path"] = path
        offloaded["is_file"] = is_file

    async def fake_to_thread(func, *args, **kwargs):
        offloaded["func"] = func
        return func(*args, **kwargs)

    monkeypatch.setattr(open_in_api.sys, "platform", "win32")
    monkeypatch.setattr(open_in_api, "_open_in_sync", fake_open_in_sync)
    monkeypatch.setattr(open_in_api.asyncio, "to_thread", fake_to_thread)

    response = await open_in_api.open_in(
        open_in_api.OpenInRequest(app="finder", path=str(tmp_path))
    )

    assert response.ok is True
    assert offloaded["func"] is fake_open_in_sync
    assert offloaded["path"] == tmp_path.resolve()
    assert offloaded["is_file"] is False


def test_a_quote_in_the_path_stays_data_in_the_applescript(monkeypatch, tmp_path) -> None:
    """A double quote is legal in a macOS filename, and this is osascript.

    Interpolated raw, a directory named `x" & (do shell script "…") & "` closed
    the AppleScript string literal and the rest of its own name ran as code —
    with the path coming from a POST body.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(open_in_api, "_run_command", lambda args: calls.append(args))

    hostile = tmp_path / 'x" & (do shell script "echo pwned") & "'
    hostile.mkdir()
    open_in_api._open_terminal(hostile)

    script = calls[0][-1]
    # The payload survives as a quoted literal: every quote it contains is
    # escaped, so none of it can terminate the string it sits in.
    assert '\\"' in script
    assert 'do shell script \\"echo pwned\\"' in script


def test_applescript_literals_round_trip_through_osascript(tmp_path) -> None:
    """The escaping is only right if osascript itself gives the path back."""
    osascript = shutil.which("osascript")
    if osascript is None:
        pytest.skip("osascript is macOS-only")

    for name in [
        "plain",
        'a"b',
        "a\\b",
        # A backslash immediately before a quote is where a wrong replacement
        # order silently corrupts the literal — and re-opens the injection.
        'a\\"b',
        "a\nb",
        "a\rb",
        'x" & (do shell script "echo pwned") & "',
    ]:
        literal = open_in_api._applescript_string(str(tmp_path / name))
        result = subprocess.run(
            [osascript, "-e", f"return {literal}"],
            capture_output=True,
            text=True,
            check=True,
        )
        # CR and LF compare equal here: osascript hands a carriage return back
        # as a newline. What matters is that the payload came back as text at
        # all rather than being executed, and that nothing was lost.
        expected = str(tmp_path / name).replace("\r", "\n")
        assert result.stdout.strip().replace("\r", "\n") == expected


def test_a_windows_shell_metacharacter_in_the_path_is_refused(monkeypatch, tmp_path) -> None:
    """cmd.exe re-parses what `cmd /c start` hands it.

    The AppleScript side was quoted; this one cannot be — cmd's quoting does
    not survive being nested inside `start` — so a path it would read as
    syntax is refused instead. `x & calc` is a legal directory name, and the
    agent can create it.
    """
    monkeypatch.setattr(open_in_api, "_run_command", lambda args: None)

    with pytest.raises(HTTPException) as raised:
        open_in_api._open_windows_app("cursor", tmp_path / "x & calc")

    assert raised.value.status_code == 400

    with pytest.raises(HTTPException):
        open_in_api._open_windows_terminal(tmp_path / 'x" & calc')


def test_an_ordinary_windows_path_still_opens(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(open_in_api, "_run_command", lambda args: calls.append(args))

    open_in_api._open_windows_app("cursor", tmp_path / "project")

    assert calls == [["cmd", "/c", "start", "", "cursor", str(tmp_path / "project")]]
