"""SSHKaos behaviour that does not need a server to check.

The rest of the SSH suite talks to a real host and is skipped almost
everywhere; these three are pure logic over the SFTP client, so a stand-in
is enough — and each of them was wrong.
"""

from __future__ import annotations

from typing import Any

import pytest

from kaos.ssh import SSHKaos


class _FakeFile:
    def __init__(self, text: str) -> None:
        self._text = text

    async def __aenter__(self) -> "_FakeFile":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def read(self) -> str:
        return self._text


class _FakeSFTP:
    def __init__(self, text: str = "", existing: set[str] | None = None) -> None:
        self._text = text
        self._existing = existing or set()
        self.made: list[str] = []

    def open(self, *_args: object, **_kwargs: object) -> _FakeFile:
        return _FakeFile(self._text)

    async def exists(self, path: str) -> bool:
        return path in self._existing

    async def mkdir(self, path: str) -> None:
        if path in self._existing:
            raise OSError("SFTPFailure: file already exists")
        self._existing.add(path)
        self.made.append(path)


def _kaos(sftp: Any) -> SSHKaos:
    return SSHKaos(connection=None, sftp=sftp, home="/home/u", cwd="/home/u", host="h")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_readlines_keeps_the_newlines() -> None:
    """The Read tool joins these back with "" and expects the breaks intact."""
    kaos = _kaos(_FakeSFTP("alpha\nbeta\ngamma\n"))

    lines = [line async for line in kaos.readlines("/f")]

    assert lines == ["alpha\n", "beta\n", "gamma\n"]
    assert "".join(lines) == "alpha\nbeta\ngamma\n"


@pytest.mark.asyncio
async def test_mkdir_on_an_existing_directory_with_exist_ok_is_a_no_op() -> None:
    """It used to fall through and let the server raise SFTPFailure."""
    sftp = _FakeSFTP(existing={"/home/u/there"})
    kaos = _kaos(sftp)

    await kaos.mkdir("/home/u/there", exist_ok=True)

    assert sftp.made == []


@pytest.mark.asyncio
async def test_mkdir_without_exist_ok_still_refuses() -> None:
    kaos = _kaos(_FakeSFTP(existing={"/home/u/there"}))

    with pytest.raises(FileExistsError):
        await kaos.mkdir("/home/u/there")


@pytest.mark.asyncio
async def test_create_closes_the_connection_when_the_cwd_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad `cwd` is the common case, and it used to leak the connection."""
    import asyncssh

    closed: list[bool] = []

    class _Connection:
        def close(self) -> None:
            closed.append(True)

        async def start_sftp_client(self) -> Any:
            return _SFTP()

    class _SFTP:
        async def realpath(self, _p: str) -> str:
            return "/home/u"

        async def chdir(self, _p: str) -> None:
            raise FileNotFoundError("no such directory")

    async def _connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(asyncssh, "connect", _connect)

    with pytest.raises(FileNotFoundError):
        await SSHKaos.create("host", cwd="/nope")

    assert closed == [True]
