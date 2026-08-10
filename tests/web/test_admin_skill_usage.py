from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kimi_cli.analytics import skill_usage
from kimi_cli.metadata import Metadata, WorkDirMeta, save_metadata
from kimi_cli.skill.manager import SkillManager
from kimi_cli.web.api import admin
from kimi_cli.web.user_auth import require_admin

DAY = 86400.0


@pytest.fixture(autouse=True)
def _clear_caches():
    """The aggregator has a 60s response cache and a per-file memo; without
    this, results leak between tests."""
    skill_usage.reset_cache()
    yield
    skill_usage.reset_cache()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _wire(path: Path, records: list[tuple[float, str, dict[str, Any]]]) -> None:
    """Write a wire.jsonl, metadata header first (as the real writer does)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "metadata", "protocol_version": "1"})]
    for ts, msg_type, payload in records:
        lines.append(json.dumps({"timestamp": ts, "message": {"type": msg_type, "payload": payload}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(args)},
        "extras": None,
    }


def _read_file(call_id: str, path: str) -> dict[str, Any]:
    return _tool_call(call_id, "ReadFile", {"path": path})


def _tool_result(call_id: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"tool_call_id": call_id, "return_value": {"is_error": is_error}}


def _turn(text: str) -> dict[str, Any]:
    """TurnBegin with the list-form user_input that real logs actually use."""
    return {"user_input": [{"type": "text", "text": text}]}


def _sub(
    inner_type: str,
    inner_payload: dict[str, Any],
    subagent_type: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"event": {"type": inner_type, "payload": inner_payload}}
    if subagent_type is not None:
        payload["subagent_type"] = subagent_type
        payload["agent_id"] = f"{subagent_type}-1"
    return payload


def _make_env(
    monkeypatch,
    tmp_path: Path,
    *,
    builtin_skills: dict[str, str] | None = None,
) -> tuple[Path, SkillManager]:
    """Set up an isolated share dir, one work dir, and a SkillManager.

    Returns ``(sessions_root, manager)``.
    """
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    work_dir = tmp_path / "project"
    work_dir.mkdir(parents=True, exist_ok=True)
    metadata = Metadata(work_dirs=[WorkDirMeta(path=str(work_dir))])
    save_metadata(metadata)

    builtin = tmp_path / "builtin"
    writable = tmp_path / "writable"
    builtin.mkdir(exist_ok=True)
    for dir_name, frontmatter_name in (builtin_skills or {}).items():
        d = builtin / dir_name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {frontmatter_name}\ndescription: Test\n---\n", encoding="utf-8"
        )

    manager = SkillManager(builtin, writable)
    monkeypatch.setattr(admin, "_skill_manager", lambda: manager)
    return metadata.work_dirs[0].sessions_dir, manager


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(admin.router)
    app.dependency_overrides[require_admin] = lambda: {"id": "admin", "role": "admin"}
    return TestClient(app)


def _by_name(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for entry in report["skills"]:
        if entry["name"] == name:
            return entry
    return None


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_top_level_skill_md_read_is_counted(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [(now, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md"))],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["read_count"] == 1
    assert entry["count"] == 1
    assert entry["installed"] is True
    assert report["totals"]["invocations"] == 1


def test_read_nested_two_levels_in_subagent_events_is_counted(
    monkeypatch, tmp_path: Path
) -> None:
    """69% of real tool calls happen inside subagents; missing the recursion
    would silently drop the majority of usage."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    nested = _sub("SubagentEvent", _sub("ToolCall", _read_file("c1", "/skills/alpha/SKILL.md")))
    _wire(sessions / "s1" / "wire.jsonl", [(now, "SubagentEvent", nested)])

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["read_count"] == 1


def test_unrelated_markdown_read_is_not_counted(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path)
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", _read_file("c1", "/tmp/random.md")),
            (now, "ToolCall", _read_file("c2", "/home/me/notes/README.md")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    assert report["skills"] == []
    assert report["totals"]["invocations"] == 0


def test_flat_md_counted_only_inside_a_skills_root(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path)
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            # Inside a recognised skills root -> counted.
            (now, "ToolCall", _read_file("c1", "/home/me/.kimi/skills/gamma.md")),
            # Same basename outside any skills root -> ignored.
            (now, "ToolCall", _read_file("c2", "/home/me/docs/gamma.md")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "gamma")
    assert entry is not None
    assert entry["read_count"] == 1


def test_directory_name_is_aliased_to_frontmatter_name(monkeypatch, tmp_path: Path) -> None:
    """The path yields the directory name, the admin card shows the frontmatter
    name. Without the alias table these silently fail to join."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"docx": "DOCX Pro"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [(now, "ToolCall", _read_file("c1", "/skills/docx/SKILL.md"))],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "DOCX Pro")
    assert entry is not None
    assert entry["key"] == "docx"
    assert entry["installed"] is True
    # Matches what /api/admin/skills reports for the same skill.
    assert entry["name"] == manager.list_skills()[0].name


def test_deleted_skill_still_reported_as_not_installed(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path)  # no skills installed
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [(now, "ToolCall", _read_file("c1", "/skills/zeta/SKILL.md"))],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "zeta")
    assert entry is not None
    assert entry["installed"] is False
    assert entry["count"] == 1


def test_error_result_is_attributed_back_to_the_skill(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md")),
            (now, "ToolResult", _tool_result("c1", is_error=True)),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["error_count"] == 1


def test_resource_reads_tracked_separately_from_count(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md")),
            (now, "ToolCall", _read_file("c2", "/skills/alpha/references/deep.md")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["count"] == 1, "resource reads must not inflate the headline count"
    assert entry["resource_read_count"] == 1


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------


def test_builtin_dir_classifies_as_builtin(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    monkeypatch.setattr(
        "kimi_cli.skill.get_builtin_skills_dir", lambda: tmp_path / "builtin"
    )
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [(now, "ToolCall", _read_file("c1", str(tmp_path / "builtin/alpha/SKILL.md")))],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    assert _by_name(report, "alpha")["source"] == "builtin"


def test_custom_skills_host_path_classifies_as_extra(monkeypatch, tmp_path: Path) -> None:
    """The user-picked global library is where skills are actually read from on
    many installs; labelling it "external" reads backwards."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    global_dir = tmp_path / "global-skills"
    monkeypatch.setenv("CUSTOM_SKILLS_HOST_PATH", str(global_dir))
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [(now, "ToolCall", _read_file("c1", str(global_dir / "alpha/SKILL.md")))],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry["source"] == "extra"
    # Still joins to the installed skill for name/管理 purposes.
    assert entry["installed"] is True


def test_extra_wins_over_builtin_when_both_match(monkeypatch, tmp_path: Path) -> None:
    """The extra scope has higher discovery priority at runtime, so a user
    library shadowing a bundled skill should report as extra."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    global_dir = tmp_path / "global-skills"
    monkeypatch.setenv("CUSTOM_SKILLS_HOST_PATH", str(global_dir))
    monkeypatch.setattr(
        "kimi_cli.skill.get_builtin_skills_dir", lambda: tmp_path / "builtin"
    )
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", _read_file("c1", str(tmp_path / "builtin/alpha/SKILL.md"))),
            (now, "ToolCall", _read_file("c2", str(global_dir / "alpha/SKILL.md"))),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    assert _by_name(report, "alpha")["source"] == "extra"


def test_unknown_root_still_falls_back_to_external(monkeypatch, tmp_path: Path) -> None:
    """Without CUSTOM_SKILLS_HOST_PATH the scope cannot be recovered from a
    historic path; that degrades the label, never the count."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    monkeypatch.delenv("CUSTOM_SKILLS_HOST_PATH", raising=False)
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [(now, "ToolCall", _read_file("c1", "/somewhere/else/alpha/SKILL.md"))],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry["source"] == "external"
    assert entry["count"] == 1, "an unrecognised root must not lose the count"


# ---------------------------------------------------------------------------
# Origin attribution (main agent vs subagents)
# ---------------------------------------------------------------------------


def test_origin_separates_main_from_subagents(monkeypatch, tmp_path: Path) -> None:
    """A skill only ever auto-read by `explore` must not look like one users
    actively reach for."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md")),
            (now, "SubagentEvent",
             _sub("ToolCall", _read_file("c2", "/skills/alpha/SKILL.md"), "explore")),
            (now, "SubagentEvent",
             _sub("ToolCall", _read_file("c3", "/skills/alpha/SKILL.md"), "explore")),
            (now, "SubagentEvent",
             _sub("ToolCall", _read_file("c4", "/skills/alpha/SKILL.md"), "coder")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["count"] == 4
    assert entry["by_origin"] == {"explore": 2, "main": 1, "coder": 1}
    assert report["totals"]["by_origin"] == {"explore": 2, "main": 1, "coder": 1}
    assert sum(entry["by_origin"].values()) == entry["count"]


def test_nested_subagent_attributes_to_innermost(monkeypatch, tmp_path: Path) -> None:
    """The innermost subagent is the one that actually issued the call."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    nested = _sub(
        "SubagentEvent",
        _sub("ToolCall", _read_file("c1", "/skills/alpha/SKILL.md"), "coder"),
        "explore",
    )
    _wire(sessions / "s1" / "wire.jsonl", [(now, "SubagentEvent", nested)])

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["by_origin"] == {"coder": 1}


def test_subagent_without_type_falls_back_to_generic_label(
    monkeypatch, tmp_path: Path
) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [(now, "SubagentEvent", _sub("ToolCall", _read_file("c1", "/skills/alpha/SKILL.md")))],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["by_origin"] == {"subagent": 1}


def test_origin_excludes_errors_and_resource_reads(monkeypatch, tmp_path: Path) -> None:
    """by_origin must sum to `count`; errors and resource reads are facets of an
    invocation, not extra ones."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md")),
            (now, "ToolResult", _tool_result("c1", is_error=True)),
            (now, "ToolCall", _read_file("c2", "/skills/alpha/references/deep.md")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["count"] == 1
    assert entry["error_count"] == 1
    assert entry["resource_read_count"] == 1
    assert entry["by_origin"] == {"main": 1}


def test_slash_commands_are_attributed_to_main(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(sessions / "s1" / "wire.jsonl", [(now, "TurnBegin", _turn("/skill:alpha"))])

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["by_origin"] == {"main": 1}


def test_origin_respects_the_window(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md")),
            (now - 10 * DAY, "SubagentEvent",
             _sub("ToolCall", _read_file("c2", "/skills/alpha/SKILL.md"), "explore")),
        ],
    )

    wide = skill_usage.build_skill_usage(days=30, manager=manager, refresh=True)
    assert _by_name(wide, "alpha")["by_origin"] == {"main": 1, "explore": 1}

    narrow = skill_usage.build_skill_usage(days=2, manager=manager, refresh=True)
    assert _by_name(narrow, "alpha")["by_origin"] == {"main": 1}


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def test_slash_skill_command_is_counted(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(sessions / "s1" / "wire.jsonl", [(now, "TurnBegin", _turn("/skill:alpha do the thing"))])

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["slash_count"] == 1
    assert entry["read_count"] == 0, "slash invocations must not inflate read_count"
    assert report["totals"]["slash_invocations"] == 1


def test_slash_flow_command_is_counted(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"beta": "beta"})
    now = time.time()
    _wire(sessions / "s1" / "wire.jsonl", [(now, "TurnBegin", _turn("/flow:beta"))])

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "beta")
    assert entry is not None
    assert entry["slash_count"] == 1
    assert entry["flow_count"] == 1


def test_string_form_user_input_is_handled(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(sessions / "s1" / "wire.jsonl", [(now, "TurnBegin", {"user_input": "/skill:alpha"})])

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["slash_count"] == 1


def test_plain_prompt_is_not_a_slash_invocation(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path)
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "TurnBegin", _turn("please use the alpha skill")),
            (now, "TurnBegin", _turn("/compact")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    assert report["totals"]["slash_invocations"] == 0


def test_slash_display_name_folds_onto_directory_key(monkeypatch, tmp_path: Path) -> None:
    """Slash commands carry the display name, paths carry the directory name;
    both must land in one bucket."""
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"docx": "DOCX-Pro"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", _read_file("c1", "/skills/docx/SKILL.md")),
            (now, "TurnBegin", _turn("/skill:DOCX-Pro")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    assert len(report["skills"]) == 1
    entry = report["skills"][0]
    assert entry["read_count"] == 1
    assert entry["slash_count"] == 1
    assert entry["count"] == 2


# ---------------------------------------------------------------------------
# Windowing, robustness, aggregation
# ---------------------------------------------------------------------------


def test_window_excludes_old_events(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [(now - 10 * DAY, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md"))],
    )

    within = skill_usage.build_skill_usage(days=30, manager=manager, refresh=True)
    assert _by_name(within, "alpha") is not None

    outside = skill_usage.build_skill_usage(days=1, manager=manager, refresh=True)
    assert _by_name(outside, "alpha") is None
    assert outside["totals"]["invocations"] == 0


def test_daily_series_sums_to_count_and_last_used_is_max(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    older, newer = now - 2 * DAY, now - 1 * DAY
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (older, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md")),
            (newer, "ToolCall", _read_file("c2", "/skills/alpha/SKILL.md")),
            (newer, "ToolCall", _read_file("c3", "/skills/alpha/SKILL.md")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["count"] == 3
    assert sum(entry["daily"]) == entry["count"]
    assert len(entry["daily"]) == len(report["dates"]) == 30
    assert entry["last_used"] == pytest.approx(newer)
    assert entry["first_used"] == pytest.approx(older)
    assert sum(report["daily"]) == 3


def test_counts_aggregate_across_sessions(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    for sid in ("s1", "s2"):
        _wire(
            sessions / sid / "wire.jsonl",
            [(now, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md"))],
        )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["count"] == 2
    assert entry["session_count"] == 2
    assert report["totals"]["distinct_sessions"] == 2


def test_malformed_lines_are_skipped(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    wire = sessions / "s1" / "wire.jsonl"
    wire.parent.mkdir(parents=True, exist_ok=True)
    good = json.dumps(
        {
            "timestamp": now,
            "message": {"type": "ToolCall", "payload": _read_file("c1", "/skills/alpha/SKILL.md")},
        }
    )
    wire.write_text(
        "\n".join(
            [
                json.dumps({"type": "metadata", "protocol_version": "1"}),
                "{not json at all",
                good,
                '{"timestamp": 1, "message": {"type": "ToolCall", "payl',  # torn final line
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with _client() as client:
        response = client.get("/api/admin/skills/usage?days=30&refresh=true")

    assert response.status_code == 200
    entry = _by_name(response.json(), "alpha")
    assert entry is not None
    assert entry["count"] == 1


def test_bad_tool_arguments_are_skipped(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path)
    now = time.time()
    _wire(
        sessions / "s1" / "wire.jsonl",
        [
            (now, "ToolCall", {"type": "function", "id": "c1",
                               "function": {"name": "ReadFile", "arguments": "{broken"}}),
            (now, "ToolCall", {"type": "function", "id": "c2",
                               "function": {"name": "ReadFile", "arguments": "{}"}}),
            (now, "ToolCall", {"type": "function", "id": "c3",
                               "function": {"name": "ReadFile", "arguments": '{"path": 42}'}}),
            # Single-segment path is unattributable.
            (now, "ToolCall", _read_file("c4", "SKILL.md")),
        ],
    )

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    assert report["skills"] == []


def test_owner_id_drives_user_attribution(monkeypatch, tmp_path: Path) -> None:
    sessions, manager = _make_env(monkeypatch, tmp_path, builtin_skills={"alpha": "alpha"})
    now = time.time()
    session_dir = sessions / "s1"
    _wire(session_dir / "wire.jsonl", [(now, "ToolCall", _read_file("c1", "/skills/alpha/SKILL.md"))])
    (session_dir / "state.json").write_text(json.dumps({"owner_id": "user-42"}), encoding="utf-8")

    report = skill_usage.build_skill_usage(days=30, manager=manager)

    entry = _by_name(report, "alpha")
    assert entry is not None
    assert entry["user_count"] == 1
    assert report["totals"]["distinct_users"] == 1
    assert [u["user_id"] for u in report["top_users"]] == ["user-42"]


def test_empty_install_returns_zeroed_report(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    save_metadata(Metadata(work_dirs=[]))
    manager = SkillManager(tmp_path / "builtin", tmp_path / "writable")
    monkeypatch.setattr(admin, "_skill_manager", lambda: manager)

    with _client() as client:
        response = client.get("/api/admin/skills/usage?days=7&refresh=true")

    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["invocations"] == 0
    assert body["skills"] == []
    assert len(body["dates"]) == 7
    assert body["daily"] == [0] * 7


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------


def test_endpoint_clamps_days_and_caches(monkeypatch, tmp_path: Path) -> None:
    _make_env(monkeypatch, tmp_path)

    with _client() as client:
        clamped = client.get("/api/admin/skills/usage?days=9999")
        assert clamped.status_code == 200
        assert clamped.json()["window_days"] == 90

        first = client.get("/api/admin/skills/usage?days=30").json()
        second = client.get("/api/admin/skills/usage?days=30").json()
        assert second["scanned"]["cached"] is True
        assert second["generated_at"] == first["generated_at"]

        refreshed = client.get("/api/admin/skills/usage?days=30&refresh=true").json()
        assert refreshed["scanned"]["cached"] is False
        assert refreshed["generated_at"] >= first["generated_at"]


def test_usage_route_is_not_shadowed_by_the_skill_name_route(
    monkeypatch, tmp_path: Path
) -> None:
    """`/skills/usage` must resolve to the stats endpoint, not be parsed as a
    skill named "usage"."""
    _make_env(monkeypatch, tmp_path)

    with _client() as client:
        response = client.get("/api/admin/skills/usage")

    assert response.status_code == 200
    assert "totals" in response.json()
