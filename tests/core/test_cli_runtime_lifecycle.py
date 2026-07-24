from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from kimi_cli.cli import Reload, SwitchToVis, SwitchToWeb, _CLIInstanceOwner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("SessionStart hook failed"),
        asyncio.CancelledError(),
        Reload(),
        SwitchToWeb(),
        SwitchToVis(),
    ],
)
async def test_owned_cli_instance_always_closes_runtime(failure: BaseException) -> None:
    instance = AsyncMock()
    owner = _CLIInstanceOwner(instance)  # type: ignore[arg-type]

    with pytest.raises(type(failure)):
        try:
            raise failure
        finally:
            await owner.close()

    instance.close.assert_awaited_once_with()
    await owner.close()
    instance.close.assert_awaited_once_with()
