"""Shared protection against bypassing the managed Wiki mutation boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kosong.tooling import ToolError

if TYPE_CHECKING:
    from kaos.path import KaosPath

    from kimi_cli.soul.agent import Runtime


def reject_managed_wiki_target(path: KaosPath, runtime: Runtime) -> ToolError | None:
    """Reject any local descendant or symlink alias of the managed Wiki root.

    The Wiki root is intentionally not an ordinary workspace.  This check occurs
    before diff rendering or approval so no file-tool code can obtain a generic
    filesystem approval for a managed Wiki mutation.
    """
    manager = getattr(runtime, "wiki", None)
    if manager is None:
        return None
    try:
        target = Path(str(path)).expanduser().resolve(strict=False)
        root = manager.layout.root.resolve(strict=True)
        if target.is_relative_to(root):
            return ToolError(
                message="Managed Wiki files can only be changed through the Wiki tool.",
                brief="Use Wiki tool",
            )
    except OSError:
        # Let the normal file tool surface path/remote filesystem errors.  A path
        # we cannot prove is local must not be accidentally treated as the root.
        return None
    return None
