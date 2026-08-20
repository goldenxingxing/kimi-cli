"""The re-export layer has to keep up with the package it re-exports.

`kimi_cli.memory` names every symbol it forwards, which keeps the seam visible
— reading the module tells you what this application uses. The cost is that a
symbol added to Amem is not forwarded until someone regenerates the list, and
nothing says so: the import fails much later, in whatever first needs it. That
happened within a day of the split, on a threshold added upstream.

So the list is not generated at import time — it is checked here.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

#: Modules that are pure re-exports. `archivist` is not one — it keeps this
#: application's own orchestration — and `paths` and `knowledge` have no
#: counterpart upstream at all.
FORWARDED = (
    "candidates",
    "condense",
    "consolidate",
    "dedup",
    "entry",
    "recent",
    "search",
    "storage",
)


def _public_names(module_file: Path) -> set[str]:
    """Top-level public symbols defined in a source file."""
    names: set[str] = set()
    for node in ast.parse(module_file.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {n for n in names if not n.startswith("_")}


@pytest.mark.parametrize("name", FORWARDED)
def test_everything_public_upstream_is_forwarded(name: str) -> None:
    upstream = importlib.import_module(f"amem.{name}")
    local = importlib.import_module(f"kimi_cli.memory.{name}")

    expected = _public_names(Path(upstream.__file__))
    missing = sorted(n for n in expected if not hasattr(local, n))

    assert not missing, (
        f"amem.{name} defines {missing} and kimi_cli.memory.{name} does not forward them — "
        "regenerate the re-export list"
    )


@pytest.mark.parametrize("name", FORWARDED)
def test_nothing_is_forwarded_that_upstream_no_longer_has(name: str) -> None:
    """The other direction: a removed symbol should fail here, not at import."""
    local = importlib.import_module(f"kimi_cli.memory.{name}")

    for exported in getattr(local, "__all__", []):
        assert hasattr(local, exported), f"{name}.__all__ names {exported}, which is not there"


@pytest.mark.parametrize("name", FORWARDED)
def test_the_forwarded_object_is_the_upstream_one(name: str) -> None:
    """Re-export, not reimplementation — the thing this split exists to guarantee."""
    upstream = importlib.import_module(f"amem.{name}")
    local = importlib.import_module(f"kimi_cli.memory.{name}")

    for exported in getattr(local, "__all__", []):
        assert getattr(local, exported) is getattr(upstream, exported), (
            f"kimi_cli.memory.{name}.{exported} is a different object from amem.{name}.{exported}"
        )
