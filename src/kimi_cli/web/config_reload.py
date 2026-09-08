"""Applying a configuration change to a live server, without restarting it.

The desktop app writes its settings straight to `.env` and `config.toml` and
then had only one way to make them take effect: restart the whole server. That
kills every session, including one mid-turn, to deliver a changed API key.

So the app touches a signal file instead and the server does the smallest thing
that actually applies the change:

* Re-read the env file it was started from. The server's own environment is
  frozen at spawn, and two things read it later -- ``_build_global_config``,
  which merges env providers back into ``config.toml`` on *every* request (so a
  deleted provider would be resurrected and written back to disk), and every
  worker it spawns, which inherits ``os.environ``.
* Restart the workers that are idle, and leave the busy ones alone; they carry
  the change over at their next idle moment.

Only the keys below are refreshed. A settings file is not a general licence to
rewrite this process's environment.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from kimi_cli import logger

#: Where the desktop app puts the file it touches to ask for a reload.
RELOAD_SIGNAL_NAME = "reload.signal"

#: Set by the desktop supervisor so the server can re-read what it was given.
ENV_FILE_VAR = "OPENKIMO_ENV_FILE"
APP_DATA_DIR_VAR = "OPENKIMO_APP_DATA_DIR"

#: The only variables a reload may change in this process.
#:
#: LLM keys decide which providers exist; the two web keys are read per request
#: (see ``AuthMiddleware``), so refreshing them takes effect without a restart.
#: The port is deliberately absent -- the socket is already bound, and pretending
#: otherwise would be worse than saying it needs a restart.
RELOADABLE_KEYS: frozenset[str] = frozenset(
    {
        "LLM_PROVIDERS",
        "LLM_DEFAULT_PROVIDER",
        "LLM_PROVIDER",
        "LLM_THINKING",
        "LLM_TEMPERATURE",
        "KIMI_API_KEY",
        "KIMI_BASE_URL",
        "KIMI_MODEL_NAME",
        "KIMI_MODEL_MAX_CONTEXT_SIZE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "KIMI_WEB_SESSION_TOKEN",
        "KIMI_WEB_LAN_ONLY",
    }
)

#: How often the signal file is checked. One stat() of one path; the latency
#: matters more than the cost, since a person is waiting on it.
POLL_INTERVAL_S = 2.0


def reload_signal_path() -> Path | None:
    """The file to watch, or nothing when not running under the desktop app."""
    data_dir = os.environ.get(APP_DATA_DIR_VAR)
    if not data_dir:
        return None
    return Path(data_dir) / RELOAD_SIGNAL_NAME


def parse_env_file(text: str) -> dict[str, str]:
    """Read ``KEY=value`` lines the way the desktop app writes them."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def refresh_environment() -> set[str]:
    """Re-read the env file into this process. Returns the keys that moved.

    A key the file no longer sets is removed rather than left behind: leaving it
    is how a provider the user deleted stays alive in the merge.
    """
    env_file = os.environ.get(ENV_FILE_VAR)
    if not env_file:
        return set()
    try:
        values = parse_env_file(Path(env_file).read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning(f"Config reload could not read {env_file}: {exc}")
        return set()
    changed: set[str] = set()
    for key in RELOADABLE_KEYS:
        new = values.get(key)
        old = os.environ.get(key)
        if new is None or new == "":
            if old is not None:
                del os.environ[key]
                changed.add(key)
        elif new != old:
            os.environ[key] = new
            changed.add(key)
    return changed


#: The spellings app._load_env_flag accepts for a boolean setting.
_TRUTHY = {"1", "true", "yes", "on"}


def _read_settings_values() -> dict[str, str]:
    """What the settings file itself says, or {} when it cannot be read.

    Read again rather than through os.environ: refresh_environment() deletes a
    key the file sets to empty *and* one the file never mentions, and those two
    have to be told apart here — one is "turn this off", the other is "not my
    business".
    """
    env_file = os.environ.get(ENV_FILE_VAR)
    if not env_file:
        return {}
    try:
        return parse_env_file(Path(env_file).read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning(f"Config reload could not re-read {env_file}: {exc}")
        return {}


def _refresh_app_state(app) -> None:  # noqa: ANN001 - FastAPI app, kept untyped to avoid a cycle
    """Point the per-request auth checks at the refreshed environment.

    What the settings file says replaces the value, including an explicit empty
    token, which means the same thing at a reload as it does at startup: auth
    off. What the file does not mention leaves the value alone — that is the
    part that used to be wrong. A settings file that simply never named
    KIMI_WEB_SESSION_TOKEN set app.state.session_token to None here and turned
    the middleware's token check off for the rest of the process: a save that
    opened the server up.

    So these two are deliberately not symmetric with refresh_environment(),
    which drops a key the file no longer sets: turning auth or LAN-only *off*
    at a reload means writing the key empty, not deleting the line. Deleting it
    keeps the setting until the next restart, which is the direction that
    cannot surprise anyone badly.
    """
    values = _read_settings_values()

    if "KIMI_WEB_SESSION_TOKEN" in values:
        app.state.session_token = values["KIMI_WEB_SESSION_TOKEN"] or None

    # The same spellings create_app() accepts (see app._load_env_flag). Reading
    # only "true" here meant a deployment configured with `KIMI_WEB_LAN_ONLY=1`
    # — which is what run_web_server writes — started LAN-restricted and
    # stopped being so at the first reload signal.
    if "KIMI_WEB_LAN_ONLY" in values:
        app.state.lan_only = values["KIMI_WEB_LAN_ONLY"].strip().lower() in _TRUTHY


async def apply_reload(app, runner) -> None:  # noqa: ANN001 - see above
    """Refresh this process, then hand the change to the sessions that can take it."""
    changed = await asyncio.to_thread(refresh_environment)
    if changed:
        logger.info(f"Config reload refreshed: {', '.join(sorted(changed))}")
        _refresh_app_state(app)
    summary = await runner.restart_running_workers(reason="config_reload", force=False)
    logger.info(
        f"Config reload restarted {len(summary.restarted_session_ids)} idle session(s); "
        f"{len(summary.skipped_busy_session_ids)} busy session(s) will pick it up when idle"
    )


async def watch_for_reload(app, runner) -> None:  # noqa: ANN001 - see above
    """Apply a reload whenever the signal file is touched.

    Polling rather than watching: this is one stat() of one path, and a watcher
    would have to be told apart from the writes the server itself makes to
    ``config.toml`` during a normal request.
    """
    path = reload_signal_path()
    if path is None:
        return
    last = _signal_stamp(path)
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)
        stamp = _signal_stamp(path)
        if stamp == last:
            continue
        last = stamp
        try:
            await apply_reload(app, runner)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Config reload failed")


def _signal_stamp(path: Path) -> tuple[float, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime, stat.st_size)
