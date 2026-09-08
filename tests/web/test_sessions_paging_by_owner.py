"""Paging has to happen inside the caller's own sessions, not across everyone's.

The owner filter used to run on the page that came back, so `limit=2` returned
however many of the caller's sessions happened to fall inside the global first
two — their history looked truncated, and later offsets skipped other people's
sessions rather than their own.
"""

from __future__ import annotations

from types import SimpleNamespace

from kimi_cli.web.store import sessions as store


def _entry(owner_id: str | None) -> SimpleNamespace:
    # `title` is read by the page's own "fill in missing titles" pass.
    return SimpleNamespace(title="Titled", state=SimpleNamespace(archived=False, owner_id=owner_id))


def test_the_page_is_cut_after_the_owner_filter(monkeypatch) -> None:
    # Interleaved so that a page taken before filtering would lose most of
    # mine: 6 sessions, only 2 of them mine, and they are 3rd and 6th.
    entries = [
        _entry("theirs"),
        _entry("theirs"),
        _entry("mine"),
        _entry("theirs"),
        _entry("theirs"),
        _entry("mine"),
    ]
    monkeypatch.setattr(store, "_load_sessions_index_cached", lambda: entries)
    monkeypatch.setattr(store, "_ensure_title", lambda entry, refresh=False: None)
    monkeypatch.setattr(store, "_build_joint_session", lambda entry: entry)

    page = store.load_sessions_page(limit=2, offset=0, owner_filter=lambda owner: owner == "mine")

    assert [e.state.owner_id for e in page] == ["mine", "mine"]


def test_offset_walks_the_callers_own_sessions(monkeypatch) -> None:
    entries = [_entry("theirs")] * 3 + [_entry("mine") for _ in range(3)]
    monkeypatch.setattr(store, "_load_sessions_index_cached", lambda: entries)
    monkeypatch.setattr(store, "_ensure_title", lambda entry, refresh=False: None)
    monkeypatch.setattr(store, "_build_joint_session", lambda entry: entry)

    page = store.load_sessions_page(limit=2, offset=2, owner_filter=lambda owner: owner == "mine")

    # The third of my three, not "whatever the third global session was".
    assert [e.state.owner_id for e in page] == ["mine"]


def test_no_filter_still_returns_everything(monkeypatch) -> None:
    entries = [_entry("a"), _entry(None), _entry("b")]
    monkeypatch.setattr(store, "_load_sessions_index_cached", lambda: entries)
    monkeypatch.setattr(store, "_ensure_title", lambda entry, refresh=False: None)
    monkeypatch.setattr(store, "_build_joint_session", lambda entry: entry)

    page = store.load_sessions_page(limit=10, offset=0)

    assert len(page) == 3
