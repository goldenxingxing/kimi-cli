from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

from kimi_cli.wiki.manager import WikiManager


def test_manager_close_is_thread_safe_and_idempotent(tmp_path: Path, monkeypatch) -> None:
    manager = WikiManager(tmp_path / "wiki", wal=False)
    thread_ids: list[int] = []
    original_close = manager.search_index.close

    def close_search_index() -> None:
        thread_ids.append(threading.get_ident())
        original_close()

    close = Mock(side_effect=close_search_index)
    monkeypatch.setattr(manager.search_index, "close", close)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _index: manager.close(), range(24)))

    close.assert_called_once_with()
    assert thread_ids and thread_ids[0] != threading.get_ident()
