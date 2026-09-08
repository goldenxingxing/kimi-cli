from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kaos.path import KaosPath

from kimi_cli.ui.shell.console import console
from kimi_cli.ui.shell.slash import ensure_kimi_soul, registry, shell_mode_registry
from kimi_cli.utils.export import is_sensitive_file
from kimi_cli.utils.path import sanitize_cli_path, shorten_home
from kimi_cli.wire.types import TurnBegin, TurnEnd

if TYPE_CHECKING:
    from kimi_cli.ui.shell import Shell


# ---------------------------------------------------------------------------
# /export command
# ---------------------------------------------------------------------------


@registry.command
@shell_mode_registry.command
async def export(app: Shell, args: str):
    """Export current session context to a markdown file"""
    from kimi_cli.utils.export import perform_export

    soul = ensure_kimi_soul(app)
    if soul is None:
        return

    session = soul.runtime.session
    result = await perform_export(
        history=list(soul.context.history),
        session_id=session.id,
        work_dir=str(session.work_dir),
        token_count=soul.context.token_count,
        args=args,
        default_dir=Path(str(session.work_dir)),
    )
    if isinstance(result, str):
        console.print(f"[yellow]{result}[/yellow]")
        return

    output, count = result
    from kimi_cli.telemetry import track

    track("export")
    display = shorten_home(KaosPath(str(output)))
    console.print(f"[green]Exported {count} messages to {display}[/green]")
    console.print(
        "[yellow]Note: The exported file may contain sensitive information. "
        "Please be cautious when sharing it externally.[/yellow]"
    )


# ---------------------------------------------------------------------------
# /import command
# ---------------------------------------------------------------------------


def _take_confirmation_flag(args: str) -> tuple[str, bool]:
    """Split a leading ``--yes`` / ``-y`` off the argument string."""
    rest = args.strip()
    for flag in ("--yes ", "-y "):
        if rest.startswith(flag):
            return rest[len(flag) :].strip(), True
    if rest in {"--yes", "-y"}:
        return "", True
    return rest, False


@registry.command(name="import")
@shell_mode_registry.command(name="import")
async def import_context(app: Shell, args: str):
    """Import context from a file or session ID"""
    from kimi_cli.utils.export import perform_import

    soul = ensure_kimi_soul(app)
    if soul is None:
        return

    args, confirmed = _take_confirmation_flag(args)
    target = sanitize_cli_path(args)
    if not target:
        console.print("[yellow]Usage: /import <file_path or session_id>[/yellow]")
        return

    # Before the import, not after it. The warning used to be printed once the
    # content was already merged into soul.context — in the live window sent to
    # the model, in the wire file, and in every later export — where there is
    # nothing left for the reader to decide.
    if not confirmed and is_sensitive_file(Path(target).name):
        console.print(
            f"[yellow]{shorten_home(Path(target))} looks like it may contain secrets "
            "(API keys, tokens, credentials). Importing puts its contents into this "
            "session's context permanently.[/yellow]"
        )
        console.print(f"[yellow]Re-run as [bold]/import --yes {args}[/bold] to go ahead.[/yellow]")
        return

    session = soul.runtime.session
    raw_max_context_size = (
        soul.runtime.llm.max_context_size if soul.runtime.llm is not None else None
    )
    max_context_size = (
        raw_max_context_size
        if isinstance(raw_max_context_size, int) and raw_max_context_size > 0
        else None
    )
    result = await perform_import(
        target=target,
        current_session_id=session.id,
        work_dir=session.work_dir,
        context=soul.context,
        max_context_size=max_context_size,
    )
    if isinstance(result, str):
        console.print(f"[red]{result}[/red]")
        return

    source_desc, content_len = result
    from kimi_cli.telemetry import track

    track("import")

    # Write to wire file so the import appears in session replay
    await soul.wire_file.append_message(
        TurnBegin(user_input=f"[Imported context from {source_desc}]")
    )
    await soul.wire_file.append_message(TurnEnd())

    console.print(
        f"[green]Imported context from {source_desc} "
        f"({content_len} chars) into current session.[/green]"
    )
