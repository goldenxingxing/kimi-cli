"""Tests for Environment.detect() and git-bash resolution on Windows."""

from __future__ import annotations

import platform
import subprocess
import sys

import pytest
from kaos.path import KaosPath

from kimi_cli.utils.environment import (
    Environment,
    GitBashNotFoundError,
    find_git_bash_path,
    is_windows,
)


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_linux(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "version", lambda: "5.15.0-123-generic")

    async def _mock_is_file(self: KaosPath) -> bool:
        return str(self) == "/usr/bin/bash"

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()
    assert env.os_kind == "Linux"
    assert env.os_arch == "x86_64"
    assert env.os_version == "5.15.0-123-generic"
    assert env.shell_name == "bash"
    assert str(env.shell_path) == "/usr/bin/bash"


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_linux_falls_back_to_sh(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "version", lambda: "5.15.0")

    async def _mock_is_file(self: KaosPath) -> bool:
        return False  # No bash anywhere

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()
    assert env.shell_name == "sh"
    assert str(env.shell_path) == "/bin/sh"


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_windows_with_env_override(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19044")
    monkeypatch.setenv("KIMI_CLI_GIT_BASH_PATH", r"D:\custom\bash.exe")

    async def _mock_is_file(self: KaosPath) -> bool:
        return str(self) == r"D:\custom\bash.exe"

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()
    assert env.os_kind == "Windows"
    assert env.shell_name == "bash"
    assert str(env.shell_path) == r"D:\custom\bash.exe"


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_windows_invalid_override_raises(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19044")
    monkeypatch.setenv("KIMI_CLI_GIT_BASH_PATH", r"D:\nonexistent\bash.exe")

    async def _mock_is_file(self: KaosPath) -> bool:
        return False

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    # The low-level resolver still raises — the web capabilities probe reads it.
    with pytest.raises(GitBashNotFoundError) as excinfo:
        await find_git_bash_path()

    assert "KIMI_CLI_GIT_BASH_PATH" in str(excinfo.value)
    assert "D:\\nonexistent\\bash.exe" in str(excinfo.value)

    # Detection itself degrades instead of aborting the session.
    env = await Environment.detect()
    assert env.has_shell is False
    assert env.shell_path is None
    assert "D:\\nonexistent\\bash.exe" in (env.shell_error or "")


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_windows_via_where_git(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19044")
    monkeypatch.delenv("KIMI_CLI_GIT_BASH_PATH", raising=False)

    # Simulate where.exe git -> C:\Program Files\Git\cmd\git.exe
    import shutil

    monkeypatch.setattr(
        shutil, "which", lambda exe: r"C:\Program Files\Git\cmd\git.exe" if exe == "git" else None
    )

    expected_bash = r"C:\Program Files\Git\cmd\..\bin\bash.exe"

    async def _mock_is_file(self: KaosPath) -> bool:
        return str(self) == expected_bash

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()
    assert env.shell_name == "bash"
    assert str(env.shell_path) == expected_bash


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_windows_checks_all_where_git_matches(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19044")
    monkeypatch.delenv("KIMI_CLI_GIT_BASH_PATH", raising=False)

    shim_git = r"C:\Users\me\scoop\shims\git.exe"

    def fake_run(args, **kwargs):
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        if args == [shim_git, "--exec-path"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="shim failed")
        assert args == ["where.exe", "git"]
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=shim_git + "\n" + r"C:\Program Files\Git\cmd\git.exe" + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    expected_bash = r"C:\Program Files\Git\cmd\..\bin\bash.exe"

    async def _mock_is_file(self: KaosPath) -> bool:
        return str(self) == expected_bash

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()
    assert env.shell_name == "bash"
    assert str(env.shell_path) == expected_bash


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_windows_resolves_shim_only_git(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19044")
    monkeypatch.delenv("KIMI_CLI_GIT_BASH_PATH", raising=False)

    shim_git = r"C:\Users\me\scoop\shims\git.exe"

    def fake_run(args, **kwargs):
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        if args == ["where.exe", "git"]:
            return subprocess.CompletedProcess(args, 0, stdout=shim_git + "\n", stderr="")
        if args == [shim_git, "--exec-path"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="C:/Users/me/scoop/apps/git/current/mingw64/libexec/git-core\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected subprocess args: {args!r}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    expected_bash = r"C:\Users\me\scoop\apps\git\current\bin\bash.exe"

    async def _mock_is_file(self: KaosPath) -> bool:
        return str(self) == expected_bash

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()
    assert env.shell_name == "bash"
    assert str(env.shell_path) == expected_bash


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_windows_default_install_location(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19044")
    monkeypatch.delenv("KIMI_CLI_GIT_BASH_PATH", raising=False)

    import shutil

    # Simulate `where.exe git` returning nothing
    monkeypatch.setattr(shutil, "which", lambda exe: None)

    fallback = r"C:\Program Files\Git\bin\bash.exe"

    async def _mock_is_file(self: KaosPath) -> bool:
        return str(self) == fallback

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()
    assert env.shell_name == "bash"
    assert str(env.shell_path) == fallback


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_environment_detection_windows_no_git_bash_anywhere(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19044")
    monkeypatch.delenv("KIMI_CLI_GIT_BASH_PATH", raising=False)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda exe: None)

    async def _mock_is_file(self: KaosPath) -> bool:
        return False

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()

    # No shell, but still a usable Environment: the session starts, only the
    # Shell tool is lost.
    assert env.os_kind == "Windows"
    assert env.has_shell is False
    assert env.shell_path is None
    assert env.shell_name is None
    assert "unavailable" in env.shell_description
    msg = env.shell_error or ""
    assert "Git for Windows" in msg
    assert "KIMI_CLI_GIT_BASH_PATH" in msg
    assert "KIMI_CLI_SHELL_PATH" in msg


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_find_git_bash_path_directly(monkeypatch):
    """Direct unit test for the helper, without going through Environment.detect()."""
    monkeypatch.setenv("KIMI_CLI_GIT_BASH_PATH", r"E:\git\bash.exe")

    async def _mock_is_file(self: KaosPath) -> bool:
        return str(self) == r"E:\git\bash.exe"

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    path = await find_git_bash_path()
    assert str(path) == r"E:\git\bash.exe"


def test_is_windows_reflects_platform_system(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert is_windows() is True
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert is_windows() is False
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert is_windows() is False


def test_git_detection_survives_output_it_cannot_decode(monkeypatch):
    """Regression guard for the v0.1.29 Windows worker crash.

    ``where.exe git`` answers in the host ANSI codepage — on a Chinese Windows,
    a localized "not found" line whose first byte is 0xD0. Decoding that
    strictly raises inside subprocess's own reader thread, which this function
    cannot catch: detection came back empty, the caller reported git-bash as
    missing, and the worker exited 1 before the user could type anything.
    """
    from kimi_cli.utils import environment

    captured: dict[str, object] = {}
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        gbk = "信息: 用提供的模式无法找到文件。".encode("gbk")
        # where.exe exits non-zero when it finds nothing, and prints the notice
        # on stdout in the ANSI codepage.
        return real_run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.stdout.buffer.write({gbk!r}); sys.exit(1)",
            ],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Must not raise, whatever the locale encoding happens to be.
    assert environment._where_git_executables() == []
    assert captured["errors"] == "replace"


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_windows_falls_back_to_bundled_portable_shell(monkeypatch):
    """No git-bash, but a portable sh.exe is shipped: use it rather than nothing."""
    from kimi_cli.utils import environment

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(platform, "version", lambda: "10.0.19044")
    monkeypatch.delenv("KIMI_CLI_GIT_BASH_PATH", raising=False)
    monkeypatch.delenv("KIMI_CLI_SHELL_PATH", raising=False)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda exe: None)

    bundled = KaosPath(r"C:\App\runtime\kimi_cli\deps\bin\sh.exe")

    async def _mock_is_file(self: KaosPath) -> bool:
        return False

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    async def _fake_bundled() -> KaosPath | None:
        return bundled

    monkeypatch.setattr(environment, "find_bundled_shell", _fake_bundled)

    env = await Environment.detect()

    assert env.has_shell is True
    assert env.shell_path == bundled
    # busybox ash is not bash, and the prompt must not claim otherwise.
    assert env.shell_name == "sh"
    assert env.shell_error is None


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_bundled_shell_lookup_prefers_the_share_dir(monkeypatch, tmp_path):
    """Same order as the bundled ripgrep: share dir, then the package's deps/bin."""
    from kimi_cli.utils import environment

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    share_bin = tmp_path / "share" / "bin"
    share_bin.mkdir(parents=True)
    shipped = share_bin / "sh.exe"
    shipped.write_text("", encoding="utf-8")
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    found = await environment.find_bundled_shell()

    assert found is not None
    assert str(found) == str(shipped)


async def test_bundled_shell_is_not_consulted_off_windows(monkeypatch):
    from kimi_cli.utils import environment

    monkeypatch.setattr(platform, "system", lambda: "Linux")

    assert await environment.find_bundled_shell() is None


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_shell_path_override_wins_on_any_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "version", lambda: "6.1.0")
    custom = tmp_path / "dash"
    custom.write_text("", encoding="utf-8")
    monkeypatch.setenv("KIMI_CLI_SHELL_PATH", str(custom))

    env = await Environment.detect()

    assert str(env.shell_path) == str(custom)
    assert env.shell_name == "sh"


@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping test on Windows")
async def test_unusable_shell_override_is_ignored_not_fatal(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "version", lambda: "6.1.0")
    monkeypatch.setenv("KIMI_CLI_SHELL_PATH", "/nowhere/nope")

    async def _mock_is_file(self: KaosPath) -> bool:
        return str(self) == "/bin/bash"

    monkeypatch.setattr(KaosPath, "is_file", _mock_is_file)

    env = await Environment.detect()

    assert str(env.shell_path) == "/bin/bash"
