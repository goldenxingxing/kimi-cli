import pytest
from pydantic import ValidationError

from kimi_cli.tools.memory import Params, UpdateOp


def test_memory_params_accept_json_encoded_operation_object() -> None:
    params = Params.model_validate(
        {"operation": '{"op":"update","id":"memory-1","content":"new value"}'}
    )

    assert params.operation == UpdateOp(id="memory-1", content="new value")


@pytest.mark.parametrize("operation", ['"update"', "[]", "{broken", "update"])
def test_memory_params_reject_non_object_operation_strings(operation: str) -> None:
    with pytest.raises(ValidationError):
        Params.model_validate({"operation": operation})
