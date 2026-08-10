"""Reconstruct skill usage statistics from session wire logs.

Why this is a log scan rather than instrumentation: kimi-cli has no ``Skill``
tool. Skills are advertised to the model as a name/path/description list injected
into the system prompt (``skill.format_skills_for_prompt``), and the model then
reads the body itself with the ordinary ``ReadFile`` tool. So a skill
"invocation" leaves no dedicated event — it is indistinguishable from any other
file read unless you look at the path.

Two signals are counted:

1. ``ReadFile`` whose ``path`` resolves to a skill body — ``<root>/<name>/SKILL.md``
   or a flat ``<root>/<name>.md``. This is the dominant path in practice.
2. ``/skill:<name>`` and ``/flow:<name>`` slash commands, recovered from
   ``TurnBegin.user_input``. ``KimiSoul.run`` emits ``TurnBegin`` with the raw
   user input *before* parsing the slash command, so the original text survives
   in the log. Slash invocations load the body via ``read_skill_text`` rather
   than the ``ReadFile`` tool, so the two signals never double count.

Deliberately *not* counted: ``Shell``/``Glob`` touching a skill directory. Those
fire repeatedly for a single logical use and would need de-duplication rules
that add more error than signal.

All counting is per-day, so restricting to a window is an exact sum over the
in-window days rather than an approximation.

Every invocation is also attributed to its *origin* — ``"main"`` for the
top-level agent, or the emitting subagent's type. Subagents account for the bulk
of tool traffic in a real session, so without this a skill that only ever gets
auto-read by ``explore`` looks identical to one users actively reach for.

Scaling. Current strategy is an on-demand full scan with mtime pruning, a
per-file memo, and a short-lived response cache. If session volume outgrows it:

* *Stage 2 — incremental tail reads.* ``wire.jsonl`` is strictly append-only
  (``WireFile.append_record`` opens in ``"a"`` mode), so a persisted
  ``{path: (mtime, size, offset, partial)}`` index would make steady-state cost
  proportional to newly appended bytes rather than total bytes.
* *Stage 3 — materialised table.* Blocked today: ``web/db/database.py`` has no
  migration mechanism, only ``CREATE TABLE IF NOT EXISTS`` at startup.
* *Stage 4 — real instrumentation.* Requires giving telemetry a local sink or
  adding a dedicated wire event; both are larger changes.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from kimi_cli import logger
from kimi_cli.analytics.wire_scan import WIRE_FILENAME, iter_session_dirs, iter_wire_events

__all__ = ["build_skill_usage", "attribute_path", "reset_cache"]


# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

_CACHE_TTL = 60.0
"""Response cache lifetime, mirroring vis/api/statistics.py."""

_SCAN_BUDGET_SEC = 8.0
"""Wall-clock ceiling for one scan. On overrun we return partial results marked
``truncated`` — an approximate dashboard beats a hung request."""

_MEMO_MAX = 5000
"""Per-file memo capacity (LRU). Keyed by (path, mtime_ns, size)."""

_MAX_DAYS = 90

_SKILL_MD = "skill.md"

_SLASH_RE = re.compile(r"^/(skill|flow):([A-Za-z0-9_-]+)(?=\s|$)")

# Trailing directory layouts that mark a skills root, mirroring
# skill.__init__._get_{user_brand,project_generic,project_brand}_skills_dir_candidates.
_ROOT_SUFFIXES: tuple[tuple[str, ...], ...] = (
    (".kimi", "skills"),
    (".claude", "skills"),
    (".codex", "skills"),
    (".agents", "skills"),
    ("agents", "skills"),  # ~/.config/agents/skills
)


# --------------------------------------------------------------------------
# Path attribution
# --------------------------------------------------------------------------


def _norm_parts(raw: str) -> tuple[str, ...]:
    """Split a logged path into components without touching the filesystem.

    Never resolves: the file may be long gone, and resolving would cost a
    syscall per event. Relative paths are accepted — real logs contain them.
    """
    cleaned = raw.replace("\\", "/").strip()
    if not cleaned:
        return ()
    return PurePosixPath(os.path.normpath(cleaned)).parts


def _looks_like_skills_root(parts: Iterable[str], roots: set[tuple[str, ...]]) -> bool:
    """Whether *parts* names a directory we accept as a skills root."""
    tup = tuple(parts)
    if not tup:
        return False
    if tup in roots:
        return True
    return any(len(tup) >= len(sfx) and tup[-len(sfx) :] == sfx for sfx in _ROOT_SUFFIXES)


def attribute_path(
    raw: str,
    known_roots: set[tuple[str, ...]],
    static_roots: set[tuple[str, ...]] | None = None,
) -> tuple[str, str] | None:
    """Map a ``ReadFile`` path back to a skill key.

    Returns ``(normalized_key, kind)`` where *kind* is ``"skill_md"`` or
    ``"flat"``, or ``None`` when the path is not a skill body.

    Attribution reverse-parses the path rather than consulting the live
    ``SkillManager``: historic sessions reference skills that have since been
    renamed, deleted, or that live in a project scope the admin panel never
    sees. Current install state is applied afterwards, as decoration.

    *known_roots* is mutated: Rule A teaches it the roots it discovers so that
    Rule B can recognise flat siblings later in the same scan.
    """
    parts = _norm_parts(raw)
    if len(parts) < 2:
        return None
    roots = known_roots | (static_roots or set())

    # Rule A — canonical subdirectory form <root>/<name>/SKILL.md.
    # No root check needed: SKILL.md is the spec-defined marker.
    if parts[-1].casefold() == _SKILL_MD:
        key = parts[-2].casefold()
        if not key or key.startswith("."):
            return None
        known_roots.add(parts[:-2])
        return key, "skill_md"

    # Rule B — flat form <root>/<name>.md, only inside a recognised skills root.
    # Without this guard every ReadFile on any markdown file would manufacture a
    # phantom skill.
    if parts[-1].casefold().endswith(".md") and _looks_like_skills_root(parts[:-1], roots):
        key = parts[-1][:-3].casefold()
        if key and key != "skill" and not key.startswith("."):
            return key, "flat"

    # Rule C — everything else is invisible. Not counted, not bucketed.
    return None


@dataclass(frozen=True, slots=True)
class _KnownRoots:
    """The skills roots this installation knows about, as normalised tuples."""

    builtin: tuple[str, ...] | None = None
    managed: tuple[str, ...] | None = None
    extra: frozenset[tuple[str, ...]] = frozenset()

    def all_roots(self) -> set[tuple[str, ...]]:
        out = {r for r in (self.builtin, self.managed) if r}
        out |= set(self.extra)
        return out


def _known_roots() -> _KnownRoots:
    """Resolve the configured skills roots.

    ``extra`` covers the user-picked global library: ``CUSTOM_SKILLS_HOST_PATH``
    (fed by the desktop launcher) plus ``extra_skill_dirs`` from config. On many
    installs this — not the bundled ``builtin`` copy — is where skills are
    actually read from.

    Caveat: these are resolved from the *current* process environment. A report
    generated by a server that lacks ``CUSTOM_SKILLS_HOST_PATH`` cannot tell
    that a historic path came from the extra scope, and will fall back to
    ``"external"``. That degrades a label, never a count.
    """
    builtin: tuple[str, ...] | None = None
    managed: tuple[str, ...] | None = None
    extra: set[tuple[str, ...]] = set()

    try:
        from kimi_cli.skill import get_builtin_skills_dir

        builtin = _norm_parts(str(get_builtin_skills_dir()))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not resolve builtin skills dir for usage attribution")
    try:
        from kimi_cli.skill.manager import get_managed_skill_dir

        managed = _norm_parts(str(get_managed_skill_dir()))
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not resolve managed skill dir for usage attribution")

    raw_extra: list[str] = []
    env_dir = os.environ.get("CUSTOM_SKILLS_HOST_PATH")
    if env_dir:
        raw_extra.append(env_dir)
    try:
        from kimi_cli.config import load_config

        raw_extra.extend(load_config().extra_skill_dirs or [])
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not read extra_skill_dirs for usage attribution")

    for raw in raw_extra:
        try:
            extra.add(_norm_parts(str(Path(raw).expanduser())))
        except Exception:  # pragma: no cover - defensive
            continue
    extra.discard(())

    return _KnownRoots(builtin=builtin, managed=managed, extra=frozenset(extra))


def _classify_source(roots: set[tuple[str, ...]], known: _KnownRoots) -> str:
    """Classify a skill's origin from the roots its reads came from.

    Checked most-specific first: ``extra`` wins over ``builtin`` because a user
    library commonly shadows same-named bundled skills, and the extra scope has
    the higher discovery priority at runtime.
    """
    if known.extra & roots:
        return "extra"
    if known.builtin is not None and known.builtin in roots:
        return "builtin"
    if known.managed is not None and known.managed in roots:
        return "managed"
    if roots:
        return "external"
    return "unknown"


# --------------------------------------------------------------------------
# Accumulator
# --------------------------------------------------------------------------

_COUNTERS = ("read", "slash", "flow", "error", "resource")


def _new_day() -> dict[str, int]:
    return dict.fromkeys(_COUNTERS, 0)


@dataclass
class _Agg:
    """Per-skill accumulator.

    Every counter lives in ``daily`` keyed by ``YYYY-MM-DD`` so that windowing
    is an exact sum over in-window days, never a rescale.
    """

    daily: dict[str, dict[str, int]] = field(default_factory=dict)
    origins: dict[str, dict[str, int]] = field(default_factory=dict)
    """day -> origin ("main" / subagent type) -> billable invocations that day.

    Kept per-day for the same reason as ``daily``: windowing stays an exact sum.
    """
    first_used: float | None = None
    last_used: float | None = None
    sessions: set[str] = field(default_factory=set)
    users: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)
    roots: set[tuple[str, ...]] = field(default_factory=set)

    def bump(self, counter: str, ts: float, *, day: str, origin: str | None = None) -> None:
        bucket = self.daily.get(day)
        if bucket is None:
            bucket = _new_day()
            self.daily[day] = bucket
        bucket[counter] += 1
        # Only the two counters that make up the headline count are attributed;
        # errors and resource reads would double-count the same invocation.
        if origin is not None and counter in ("read", "slash"):
            by_origin = self.origins.setdefault(day, {})
            by_origin[origin] = by_origin.get(origin, 0) + 1
        if ts > 0:
            if self.first_used is None or ts < self.first_used:
                self.first_used = ts
            if self.last_used is None or ts > self.last_used:
                self.last_used = ts

    def total(self, counter: str) -> int:
        return sum(d[counter] for d in self.daily.values())

    def origin_totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for by_origin in self.origins.values():
            for origin, count in by_origin.items():
                out[origin] = out.get(origin, 0) + count
        return out

    def merge(self, other: _Agg) -> None:
        for day, bucket in other.daily.items():
            mine = self.daily.get(day)
            if mine is None:
                self.daily[day] = dict(bucket)
            else:
                for k, v in bucket.items():
                    mine[k] += v
        for day, by_origin in other.origins.items():
            target = self.origins.setdefault(day, {})
            for origin, count in by_origin.items():
                target[origin] = target.get(origin, 0) + count
        self.sessions |= other.sessions
        self.users |= other.users
        self.paths |= other.paths
        self.roots |= other.roots
        for ts in (other.first_used, other.last_used):
            if ts is None:
                continue
            if self.first_used is None or ts < self.first_used:
                self.first_used = ts
            if self.last_used is None or ts > self.last_used:
                self.last_used = ts

    def clipped(self, keep_days: set[str]) -> _Agg | None:
        """Return a copy restricted to *keep_days*, or ``None`` if empty."""
        kept = {d: dict(b) for d, b in self.daily.items() if d in keep_days}
        if not kept or not any(sum(b.values()) for b in kept.values()):
            return None
        return _Agg(
            daily=kept,
            origins={d: dict(o) for d, o in self.origins.items() if d in keep_days},
            first_used=self.first_used,
            last_used=self.last_used,
            sessions=set(self.sessions),
            users=set(self.users),
            paths=set(self.paths),
            roots=set(self.roots),
        )


def _date_key(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return "1970-01-01"


def _line_filter(line: str) -> bool:
    """Cheap pre-parse rejection.

    Bare tokens rather than ``'"type": "ToolCall"'`` so the test is independent
    of json separator style. A false positive costs one wasted ``json.loads``;
    a false negative would silently lose data, so err wide.
    """
    return "TurnBegin" in line or "ToolCall" in line or "ToolResult" in line


def _turn_text(payload: dict[str, Any]) -> str:
    """Extract plain text from ``TurnBegin.user_input``.

    The field is ``str | list[ContentPart]``; real logs use the list form, but
    both are handled.
    """
    ui = payload.get("user_input")
    if isinstance(ui, str):
        return ui
    if isinstance(ui, list):
        chunks: list[str] = []
        for part in ui:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


def _scan_session(session_dir: Path, *, static_roots: set[tuple[str, ...]]) -> dict[str, _Agg]:
    """Aggregate one session's wire log into ``{skill_key: _Agg}``.

    Window filtering is applied by the caller so this result can be memoised
    once and reused across different ``days`` values.
    """
    per_skill: dict[str, _Agg] = defaultdict(_Agg)
    known_roots: set[tuple[str, ...]] = set()
    pending: dict[str, str] = {}  # tool_call_id -> skill key, for error attribution
    pending_day: dict[str, str] = {}
    skill_dirs: dict[tuple[str, ...], str] = {}  # skill dir prefix -> key
    session_id = session_dir.name

    for ts, ev_type, payload, origin in iter_wire_events(
        session_dir / WIRE_FILENAME, line_filter=_line_filter
    ):
        if ev_type == "ToolCall":
            fn = payload.get("function")
            if not isinstance(fn, dict) or fn.get("name") != "ReadFile":
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    continue
            if not isinstance(args, dict):
                continue
            raw_path = args.get("path")
            if not isinstance(raw_path, str):
                continue

            day = _date_key(ts)
            hit = attribute_path(raw_path, known_roots, static_roots)
            if hit is not None:
                key, kind = hit
                agg = per_skill[key]
                agg.bump("read", ts, day=day, origin=origin)
                agg.sessions.add(session_id)
                agg.paths.add(raw_path)
                parts = _norm_parts(raw_path)
                if kind == "skill_md" and len(parts) >= 2:
                    agg.roots.add(parts[:-2])
                    skill_dirs[parts[:-1]] = key
                elif len(parts) >= 1:
                    agg.roots.add(parts[:-1])
                call_id = payload.get("id")
                if isinstance(call_id, str) and call_id:
                    pending[call_id] = key
                    pending_day[call_id] = day
                continue

            # Secondary signal: reads of other files inside a skill directory
            # already identified in this session. Tracked as "depth of use" and
            # deliberately excluded from the headline count.
            parts = _norm_parts(raw_path)
            for depth in range(len(parts) - 1, 0, -1):
                owner = skill_dirs.get(parts[:depth])
                if owner is not None:
                    per_skill[owner].bump("resource", ts, day=day)
                    break

        elif ev_type == "ToolResult":
            call_id = payload.get("tool_call_id")
            if not isinstance(call_id, str):
                continue
            key = pending.pop(call_id, None)
            day = pending_day.pop(call_id, None)
            if key is None or day is None:
                continue
            rv = payload.get("return_value")
            if isinstance(rv, dict) and rv.get("is_error"):
                per_skill[key].bump("error", ts if ts > 0 else 0.0, day=day)

        elif ev_type == "TurnBegin":
            text = _turn_text(payload).strip()
            if not text.startswith("/"):
                continue
            m = _SLASH_RE.match(text)
            if m is None:
                continue
            kind, name = m.group(1), m.group(2)
            agg = per_skill[name.casefold()]
            day = _date_key(ts)
            agg.bump("slash", ts, day=day, origin=origin)
            if kind == "flow":
                agg.bump("flow", ts, day=day)
            agg.sessions.add(session_id)

    return dict(per_skill)


def _owner_id(session_dir: Path) -> str | None:
    """Read ``owner_id`` from the session's state.json.

    wire.jsonl carries no user identity, so per-user attribution has to come
    from the sibling state file. Called lazily — only for sessions that actually
    produced a skill hit.
    """
    try:
        data = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    owner = data.get("owner_id") if isinstance(data, dict) else None
    return owner if isinstance(owner, str) and owner else None


# --------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------

_response_cache: dict[int, tuple[dict[str, Any], float]] = {}
_file_memo: OrderedDict[tuple[str, int, int], dict[str, _Agg]] = OrderedDict()


def reset_cache() -> None:
    """Drop all caches. Used by tests and by ``?refresh=true``."""
    _response_cache.clear()
    _file_memo.clear()


def _memoised_scan(
    session_dir: Path, stat: os.stat_result, static_roots: set[tuple[str, ...]]
) -> dict[str, _Agg]:
    key = (str(session_dir), stat.st_mtime_ns, stat.st_size)
    cached = _file_memo.get(key)
    if cached is not None:
        _file_memo.move_to_end(key)
        return cached
    result = _scan_session(session_dir, static_roots=static_roots)
    _file_memo[key] = result
    _file_memo.move_to_end(key)
    while len(_file_memo) > _MEMO_MAX:
        _file_memo.popitem(last=False)
    return result


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def build_skill_usage(
    *,
    days: int = 30,
    refresh: bool = False,
    manager: Any = None,
) -> dict[str, Any]:
    """Aggregate skill usage across every session in the last *days* days.

    *manager* is an optional ``SkillManager``; when supplied it maps directory
    keys to display names and marks which skills are still installed. Passing it
    in (rather than constructing one here) keeps the admin test hook that
    monkeypatches ``admin._skill_manager`` effective.
    """
    days = max(1, min(int(days), _MAX_DAYS))
    started = time.time()

    if refresh:
        reset_cache()
    else:
        cached = _response_cache.get(days)
        if cached is not None and (started - cached[1]) < _CACHE_TTL:
            payload = dict(cached[0])
            payload["scanned"] = {**payload["scanned"], "cached": True}
            return payload

    today = datetime.now(tz=UTC)
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days - 1, -1, -1)]
    keep_days = set(dates)
    since_ts = started - days * 86400
    deadline = started + _SCAN_BUDGET_SEC
    known = _known_roots()
    static_roots = known.all_roots()

    totals_by_skill: dict[str, _Agg] = defaultdict(_Agg)
    user_counts: dict[str, int] = defaultdict(int)
    sessions_seen = 0
    sessions_read = 0
    truncated = False

    for _wd, session_dir in iter_session_dirs():
        sessions_seen += 1
        if time.time() > deadline:
            truncated = True
            logger.warning("Skill usage scan exceeded {budget}s budget", budget=_SCAN_BUDGET_SEC)
            break
        wire_path = session_dir / WIRE_FILENAME
        try:
            stat = wire_path.stat()
        except OSError:
            continue
        # mtime prune: an untouched session cannot contain in-window events.
        # Single biggest lever on a mature install.
        if stat.st_mtime < since_ts:
            continue
        sessions_read += 1

        per_skill = _memoised_scan(session_dir, stat, static_roots)
        if not per_skill:
            continue

        # Reading state.json costs a syscall, so defer it until we know this
        # session actually contributed something in-window.
        owner: str | None = None
        owner_looked_up = False
        for key, agg in per_skill.items():
            windowed = agg.clipped(keep_days)
            if windowed is None:
                continue
            if not owner_looked_up:
                owner = _owner_id(session_dir)
                owner_looked_up = True
            if owner:
                windowed.users.add(owner)
                user_counts[owner] += windowed.total("read") + windowed.total("slash")
            totals_by_skill[key].merge(windowed)

    report = _render(
        totals_by_skill,
        user_counts=user_counts,
        dates=dates,
        generated_at=started,
        manager=manager,
        sessions_seen=sessions_seen,
        sessions_read=sessions_read,
        truncated=truncated,
        known=known,
    )
    _response_cache[days] = (report, started)
    return dict(report)


def _render(
    totals_by_skill: dict[str, _Agg],
    *,
    user_counts: dict[str, int],
    dates: list[str],
    generated_at: float,
    manager: Any,
    known: _KnownRoots,
    sessions_seen: int,
    sessions_read: int,
    truncated: bool,
) -> dict[str, Any]:
    alias: dict[str, str] = {}
    if manager is not None:
        try:
            alias = manager.name_index()
        except Exception:  # pragma: no cover - defensive
            logger.warning("SkillManager.name_index() failed; falling back to directory names")

    # Slash commands carry the *display* name while paths carry the *directory*
    # name. Fold display names onto their directory key so both signals land in
    # the same bucket.
    display_to_key = {v.casefold(): k for k, v in alias.items()}
    folded: dict[str, _Agg] = defaultdict(_Agg)
    for key, agg in totals_by_skill.items():
        canonical = key if key in alias else display_to_key.get(key, key)
        folded[canonical].merge(agg)

    date_index = {d: i for i, d in enumerate(dates)}
    span = len(dates)
    daily_total = [0] * span
    skills: list[dict[str, Any]] = []
    total_read = 0
    total_slash = 0
    unmatched_slash = 0
    all_sessions: set[str] = set()
    all_users: set[str] = set()
    origin_total: dict[str, int] = {}

    for key, agg in folded.items():
        read = agg.total("read")
        slash = agg.total("slash")
        count = read + slash
        if count <= 0:
            continue
        installed = key in alias
        series = [0] * span
        for day, bucket in agg.daily.items():
            idx = date_index.get(day)
            if idx is None:
                continue
            hits = bucket["read"] + bucket["slash"]
            series[idx] += hits
            daily_total[idx] += hits
        total_read += read
        total_slash += slash
        if not installed and slash and not read:
            unmatched_slash += slash
        all_sessions |= agg.sessions
        all_users |= agg.users
        by_origin = agg.origin_totals()
        for origin, n in by_origin.items():
            origin_total[origin] = origin_total.get(origin, 0) + n
        skills.append(
            {
                "name": alias.get(key, key),
                "key": key,
                "installed": installed,
                "source": _classify_source(agg.roots, known),
                "count": count,
                "by_origin": dict(sorted(by_origin.items(), key=lambda kv: -kv[1])),
                "read_count": read,
                "slash_count": slash,
                "flow_count": agg.total("flow"),
                "error_count": agg.total("error"),
                "resource_read_count": agg.total("resource"),
                "session_count": len(agg.sessions),
                "user_count": len(agg.users),
                "first_used": agg.first_used,
                "last_used": agg.last_used,
                "daily": series,
                "paths": sorted(agg.paths)[:5],
            }
        )

    skills.sort(key=lambda s: (-int(s["count"]), str(s["name"]).casefold()))

    top_users = [
        {"user_id": uid, "username": _username(uid), "count": c}
        for uid, c in sorted(user_counts.items(), key=lambda kv: -kv[1])[:10]
    ]

    return {
        "generated_at": generated_at,
        "window_days": span,
        "scanned": {
            "sessions": sessions_seen,
            "sessions_read": sessions_read,
            "duration_ms": int((time.time() - generated_at) * 1000),
            "truncated": truncated,
            "cached": False,
        },
        "totals": {
            "invocations": total_read + total_slash,
            "read_invocations": total_read,
            "slash_invocations": total_slash,
            "unmatched_slash_invocations": unmatched_slash,
            "distinct_skills": len(skills),
            "distinct_sessions": len(all_sessions),
            "distinct_users": len(all_users),
            # "main" vs each subagent type. Most tool traffic in a real session
            # comes from subagents, so a skill can look popular while no user
            # ever reached for it directly.
            "by_origin": dict(sorted(origin_total.items(), key=lambda kv: -kv[1])),
        },
        "dates": dates,
        "daily": daily_total,
        "skills": skills,
        "top_users": top_users,
    }


def _username(user_id: str) -> str:
    """Resolve a user id to a display name, falling back to a short id."""
    try:
        from kimi_cli.web.db.crud import get_user_by_id
        from kimi_cli.web.db.database import get_db

        with get_db() as conn:
            user = get_user_by_id(conn, user_id)
        if user and user.get("username"):
            return str(user["username"])
    except Exception:
        logger.debug("Could not resolve username for {uid}", uid=user_id)
    return user_id[:8]
