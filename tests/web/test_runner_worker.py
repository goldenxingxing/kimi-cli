import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kaos.path import KaosPath

from kimi_cli.web.runner.worker import configure_session_environment, run_worker


def test_configure_session_environment_uses_work_directory_output(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KIMI_OUTPUT_DIR", raising=False)
    session = SimpleNamespace(
        work_dir=KaosPath.unsafe_from_local_path(tmp_path / "project"),
    )

    configure_session_environment(session)

    assert Path(str(session.work_dir), "output").is_dir()
    assert Path(__import__("os").environ["KIMI_OUTPUT_DIR"]) == Path(
        str(session.work_dir), "output"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("wire failed"), asyncio.CancelledError()])
async def test_run_worker_closes_cli_after_wire_exit(
    monkeypatch,
    tmp_path: Path,
    failure: BaseException | None,
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session = SimpleNamespace(
        dir=session_dir,
        work_dir=KaosPath.unsafe_from_local_path(tmp_path / "project"),
    )
    fake_cli = SimpleNamespace(
        run_wire_stdio=AsyncMock(side_effect=failure),
        close=AsyncMock(),
    )

    monkeypatch.setattr(
        "kimi_cli.web.runner.worker.load_session_by_id",
        lambda _session_id: SimpleNamespace(kimi_cli_session=session),
    )
    monkeypatch.setattr(
        "kimi_cli.web.runner.worker.get_global_mcp_config_file",
        lambda: tmp_path / "missing-mcp.json",
    )
    monkeypatch.setattr(
        "kimi_cli.web.runner.worker.KimiCLI.create",
        AsyncMock(return_value=fake_cli),
    )

    if failure is None:
        await run_worker(__import__("uuid").uuid4())
    else:
        with pytest.raises(type(failure)):
            await run_worker(__import__("uuid").uuid4())

    fake_cli.close.assert_awaited_once_with()
