"""What the user types in answer to the agent belongs in the transcript.

Approval feedback and free-text question answers reach the model inside a tool
result, so the conversation used to show the agent asking and then carrying on,
with the user's own words nowhere in it.
"""

from __future__ import annotations

import pytest

from kimi_cli.wire.types import QuestionItem, QuestionOption, UserReply


def _echoed(monkeypatch: pytest.MonkeyPatch, module: str) -> list[UserReply]:
    sent: list[UserReply] = []
    monkeypatch.setattr(
        f"{module}.wire_send",
        lambda msg: sent.append(msg) if isinstance(msg, UserReply) else None,
    )
    return sent


def test_approval_feedback_is_echoed_as_the_user_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kimi_cli.soul import approval as approval_module

    sent: list[UserReply] = []
    monkeypatch.setattr(
        "kimi_cli.soul.wire_send",
        lambda msg: sent.append(msg) if isinstance(msg, UserReply) else None,
    )

    approval_module._echo_user_reply("  use a migration instead  ", "approval_feedback")

    assert [(m.text, m.source) for m in sent] == [("use a migration instead", "approval_feedback")]


def test_an_empty_or_absent_reply_is_not_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.soul import approval as approval_module

    sent: list[UserReply] = []
    monkeypatch.setattr(
        "kimi_cli.soul.wire_send",
        lambda msg: sent.append(msg) if isinstance(msg, UserReply) else None,
    )

    approval_module._echo_user_reply("   ", "approval_feedback")
    approval_module._echo_user_reply("", "approval_feedback")

    assert sent == []


def test_echoing_without_a_wire_does_not_break_the_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedded use and tests have no wire; the reply still reaches the model."""
    from kimi_cli.soul import approval as approval_module

    def no_wire(_msg):
        raise AssertionError("Wire is expected to be set when soul is running")

    monkeypatch.setattr("kimi_cli.soul.wire_send", no_wire)

    approval_module._echo_user_reply("something", "approval_feedback")


def test_only_typed_question_answers_are_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Picking a listed option is a choice, not something the user said."""
    from kimi_cli.tools import ask_user as ask_user_module

    sent = _echoed(monkeypatch, "kimi_cli.tools.ask_user")
    questions = [
        QuestionItem(
            question="Which database?",
            options=[QuestionOption(label="Postgres"), QuestionOption(label="SQLite")],
        ),
        QuestionItem(
            question="Anything else?",
            options=[QuestionOption(label="No")],
        ),
    ]

    ask_user_module._echo_typed_answers(
        questions,
        {
            "Which database?": "Postgres",  # a listed option — already on screen
            "Anything else?": "use DuckDB, and keep the schema flat",  # typed
        },
    )

    assert [(m.text, m.source) for m in sent] == [
        ("use DuckDB, and keep the schema flat", "question_answer")
    ]


def test_a_user_reply_survives_the_wire_round_trip() -> None:
    """It is recorded and replayed like any other event — that is the point."""
    from kimi_cli.wire.serde import deserialize_wire_message, serialize_wire_message

    original = UserReply(text="do it differently", source="approval_feedback")
    restored = deserialize_wire_message(serialize_wire_message(original))

    assert isinstance(restored, UserReply)
    assert (restored.text, restored.source) == ("do it differently", "approval_feedback")
