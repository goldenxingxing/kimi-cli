from pathlib import Path
from types import SimpleNamespace

from kaos.path import KaosPath

from kimi_cli.web.runner.worker import configure_session_environment


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
