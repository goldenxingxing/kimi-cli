"""The re-export layer has to keep up with the package it re-exports.

`kimi_cli.memory` names every symbol it forwards, which keeps the seam visible
— reading the module tells you what this application uses. The cost is that a
symbol added to Carryover is not forwarded until someone regenerates the list, and
nothing says so: the import fails much later, in whatever first needs it. That
happened within a day of the split, on a threshold added upstream.

So the list is not generated at import time — it is checked here.
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import fields
from pathlib import Path
from typing import get_args

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
    upstream = importlib.import_module(f"carryover.{name}")
    local = importlib.import_module(f"kimi_cli.memory.{name}")

    expected = _public_names(Path(upstream.__file__))
    missing = sorted(n for n in expected if not hasattr(local, n))

    assert not missing, (
        f"carryover.{name} defines {missing} and kimi_cli.memory.{name} does not forward them — "
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
    upstream = importlib.import_module(f"carryover.{name}")
    local = importlib.import_module(f"kimi_cli.memory.{name}")

    for exported in getattr(local, "__all__", []):
        assert getattr(local, exported) is getattr(upstream, exported), (
            f"kimi_cli.memory.{name}.{exported} is a different object from carryover.{name}.{exported}"
        )


class TestTheAdvertisedActionsExist:
    """The preamble tells an agent what to call, and it is read as authoritative.

    Carryover describes what to do and leaves the calling convention to the host,
    because it used to carry this application's and was wrong in every other
    one. The cost of holding it here is that it can go stale against the tool —
    which happened immediately: the first version named an `affirm` op the tool
    did not have.
    """

    def _declared_ops(self) -> set[str]:
        import re

        from kimi_cli.soul.dynamic_injections.cross_session_memory import _ACTIONS

        spelled = " ".join(getattr(_ACTIONS, f.name) for f in fields(_ACTIONS))
        return set(re.findall(r'"op":\s*"(\w+)"', spelled))

    def _tool_ops(self) -> set[str]:
        """Every `op` the Memory tool's operation union accepts."""
        import kimi_cli.tools.memory as tool

        found: set[str] = set()
        for value in vars(tool).values():
            annotations = getattr(value, "__annotations__", None)
            if not isinstance(annotations, dict):
                continue
            op = annotations.get("op")
            found.update(get_args(op))
        return found

    def test_every_action_named_in_the_preamble_is_a_real_operation(self) -> None:
        declared = self._declared_ops()

        assert declared, "the preamble names no operations at all"
        assert declared <= self._tool_ops(), (
            f"the preamble tells the agent to call {sorted(declared - self._tool_ops())}, "
            "which the Memory tool does not accept"
        )

    def test_the_consolidation_answers_are_both_offered(self) -> None:
        """Retiring without affirming is the shape that nags: the only recordable
        answer is yes, so a no comes back every session."""
        declared = self._declared_ops()

        assert {"retire", "affirm"} <= declared


class TestEveryOperationUpstreamIsOffered:
    """A verb this package can perform and this tool does not expose is one the
    agent cannot reach.

    Found this way: carryover gained `consolidate` — replace several entries with one
    that keeps what all of them said — and the tool did not, so the answer that
    is right more often than retiring had to be done by hand outside the agent.
    """

    def test_the_tool_exposes_what_carryover_can_do(self) -> None:
        from carryover.operations import OPERATION_NAMES

        import kimi_cli.tools.memory as tool

        exposed: set[str] = set()
        for value in vars(tool).values():
            annotations = getattr(value, "__annotations__", None)
            if isinstance(annotations, dict):
                exposed.update(get_args(annotations.get("op")))

        missing = sorted(set(OPERATION_NAMES) - exposed)
        assert not missing, (
            f"carryover performs {missing} and this tool offers no way to ask for it"
        )
