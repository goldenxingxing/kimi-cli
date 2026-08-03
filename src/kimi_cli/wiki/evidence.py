"""Typed, runtime-local evidence capture for root Wiki checkpoints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit, urlunsplit

from kaos.local import local_kaos
from kosong.tooling import ToolReturnValue
from kosong.utils.typing import JsonType

from kimi_cli.tools.file.glob import Glob
from kimi_cli.tools.file.grep_local import Grep
from kimi_cli.tools.file.read import ReadFile
from kimi_cli.tools.file.read_media import ReadMediaFile
from kimi_cli.tools.file.replace import StrReplaceFile
from kimi_cli.tools.file.write import WriteFile
from kimi_cli.tools.shell import Shell
from kimi_cli.tools.web.fetch import FetchURL
from kimi_cli.tools.web.search import SearchWeb
from kimi_cli.utils.sensitive import is_sensitive_file
from kimi_cli.wiki.models import SourceRef, has_url_credentials
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.triggers import (
    EvidenceObservation,
    ProducerRole,
    WikiCheckpoint,
    WikiEvidence,
    WikiTurnCoordinator,
)
from kimi_cli.wiki.value_gate import contains_sensitive_text

if TYPE_CHECKING:
    from kimi_cli.soul.agent import Runtime

_MAX_RESULT_BYTES = 128 * 1024
_MAX_TRACKED_TURNS = 32
_MAX_TRACKED_EVIDENCE_PER_TURN = 64
_MAX_IN_FLIGHT_TOOL_CALLS = 128
_MAX_CHECKPOINT_EVIDENCE = 8
_NON_TRANSIENT_CLASSES = frozenset(
    {"workspace-file", "shell-result", "web-document", "workspace-mutation"}
)
_TRANSIENT_COMMAND = re.compile(
    r"^(?:date|pwd|ps(?:\s+.*)?|top(?:\s+.*)?|git\s+status(?:\s+.*)?)$",
    re.IGNORECASE,
)
_TEST_COUNT_ONLY = re.compile(
    r"^\s*(?:=+\s*)?(?:\d+\s+)?(?:passed|failed|skipped|errors?)"
    r"(?:\s*,\s*\d+\s+(?:passed|failed|skipped|errors?))*"
    r"(?:\s+in\s+[0-9.]+s)?\s*(?:=+)?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _TrackedEvidence:
    evidence_id: str
    source_class: str
    triggering: bool


class WikiEvidenceReporter:
    """Convert successful built-in tool results into trusted, hash-only evidence."""

    def __init__(
        self,
        coordinator: WikiTurnCoordinator,
        runtime: Runtime,
        *,
        producer_role: ProducerRole = "root",
        producer_id: str | None = None,
        run_generation: int | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._runtime = runtime
        self._producer_role: ProducerRole = producer_role
        self._producer_id = producer_id
        self._run_generation = run_generation
        self._tracked: dict[str, list[_TrackedEvidence]] = {}
        self._active_root_turn_id: str | None = None
        self._managed_turn_lifecycle = False
        self._turn_generation = 0
        self._tool_turns: dict[str, tuple[str, int]] = {}

    def start_root_turn(self, root_turn_id: str) -> None:
        """Bind capture to one accepted real root turn, never a synthetic follow-up."""
        if self._producer_role != "root" or self._runtime.role != "root":
            return
        self._managed_turn_lifecycle = True
        self._turn_generation += 1
        self._active_root_turn_id = root_turn_id
        self._tracked.setdefault(root_turn_id, [])
        while len(self._tracked) > _MAX_TRACKED_TURNS:
            self._tracked.pop(next(iter(self._tracked)))

    def finish_root_turn(self, root_turn_id: str) -> None:
        """Invalidate the turn generation after its tool tasks and completion finish."""
        if self._active_root_turn_id == root_turn_id:
            self._active_root_turn_id = None

    def begin_tool_call(self, tool_call_id: str) -> None:
        """Bind an actual tool execution to the current root turn generation."""
        root_turn_id = self._current_root_turn_id()
        if root_turn_id is None:
            return
        self._tool_turns[tool_call_id] = (root_turn_id, self._turn_generation)
        while len(self._tool_turns) > _MAX_IN_FLIGHT_TOOL_CALLS:
            self._tool_turns.pop(next(iter(self._tool_turns)))

    def abandon_tool_call(self, tool_call_id: str) -> None:
        self._tool_turns.pop(tool_call_id, None)

    async def observe(
        self,
        tool: object,
        arguments: JsonType,
        result: ToolReturnValue,
        *,
        tool_call_id: str,
    ) -> WikiEvidence | None:
        """Observe one actual built-in tool result without trusting model claims."""
        root_turn_id = self._root_turn_for_result(tool_call_id)
        if root_turn_id is None or result.is_error:
            return None

        tool_type = type(tool)
        mutation = tool_type in {WriteFile, StrReplaceFile}
        if not mutation and not _has_non_empty_output(result):
            return None

        request_hash = _arguments_hash(arguments)
        evidence: WikiEvidence | None
        if tool_type in {ReadFile, ReadMediaFile, WriteFile, StrReplaceFile}:
            evidence = await self._workspace_file(
                root_turn_id,
                cast(dict[str, JsonType], arguments) if isinstance(arguments, dict) else {},
                request_hash,
                tool_call_id,
                mutation=mutation,
            )
        elif tool_type in {Glob, Grep}:
            evidence = await self._record(
                root_turn_id=root_turn_id,
                tool_call_id=tool_call_id,
                source_class="workspace-search",
                request_hash=request_hash,
                result_hash=_result_hash(result),
                reliable=False,
                stable_snapshot=False,
                triggering=False,
            )
        elif tool_type is Shell:
            command = _string_argument(arguments, "command")
            output = _output_text(result)
            request_hash = _arguments_hash(_normalized_shell_arguments(arguments))
            result_hash = _normalized_shell_result_hash(result)
            evidence = await self._record(
                root_turn_id=root_turn_id,
                tool_call_id=tool_call_id,
                source_class="shell-result",
                request_hash=request_hash,
                result_hash=result_hash,
                source_refs=(
                    SourceRef(
                        kind="conversation",
                        session_id=self._coordinator.provenance_session_id,
                        content_hash=result_hash,
                    ),
                ),
                reliable=False,
                stable_snapshot=False,
                triggering=(
                    not _bool_argument(arguments, "run_in_background")
                    and not _is_transient_shell(command, output)
                ),
            )
        elif tool_type is SearchWeb:
            evidence = await self._record(
                root_turn_id=root_turn_id,
                tool_call_id=tool_call_id,
                source_class="web-search",
                request_hash=request_hash,
                result_hash=_result_hash(result),
                reliable=False,
                stable_snapshot=False,
                triggering=False,
            )
        elif tool_type is FetchURL:
            evidence = await self._web_document(
                root_turn_id,
                arguments,
                request_hash,
                result,
                tool_call_id,
            )
        else:
            return None

        if evidence is not None:
            records = self._tracked.setdefault(root_turn_id, [])
            if all(record.evidence_id != evidence.evidence_id for record in records):
                records.append(
                    _TrackedEvidence(
                        evidence_id=evidence.evidence_id,
                        source_class=evidence.source_class,
                        triggering=evidence.triggering,
                    )
                )
                del records[_MAX_TRACKED_EVIDENCE_PER_TURN:]
        return evidence

    async def seal_root_completion(self, conclusion: str) -> WikiCheckpoint | None:
        """Create one deduplicated root checkpoint for a reusable grounded conclusion."""
        root_turn_id = self._current_root_turn_id()
        normalized = conclusion.strip()
        if (
            root_turn_id is None
            or not _is_reusable_conclusion(normalized)
            or contains_sensitive_text(normalized)
        ):
            return None
        records = self._tracked.get(root_turn_id, ())
        triggering = tuple(record for record in records if record.triggering)
        if not triggering or not any(
            record.source_class in _NON_TRANSIENT_CLASSES for record in triggering
        ):
            return None

        summary_hash = content_hash(normalized.encode("utf-8"))
        batch = await self._coordinator.pending_batch()
        for checkpoint in batch.checkpoints:
            if checkpoint.cause == "subagent_result" and checkpoint.summary_hash == summary_hash:
                return checkpoint

        evidence_ids = tuple(record.evidence_id for record in triggering[:_MAX_CHECKPOINT_EVIDENCE])
        return await self._coordinator.create_checkpoint(
            "root_evidence",
            evidence_ids=evidence_ids,
            summary_hash=summary_hash,
        )

    def _current_root_turn_id(self) -> str | None:
        if self._producer_role != "root" or self._runtime.role != "root":
            return None
        if self._managed_turn_lifecycle:
            root_turn_id = self._active_root_turn_id
        else:
            root_turn_id = self._coordinator.active_turn_id
        if root_turn_id != self._coordinator.active_turn_id:
            return None
        return root_turn_id

    def _root_turn_for_result(self, tool_call_id: str) -> str | None:
        root_turn_id = self._current_root_turn_id()
        if not self._managed_turn_lifecycle:
            return root_turn_id
        binding = self._tool_turns.pop(tool_call_id, None)
        if binding != (root_turn_id, self._turn_generation):
            return None
        return root_turn_id

    async def _workspace_file(
        self,
        root_turn_id: str,
        arguments: dict[str, JsonType],
        request_hash: str,
        tool_call_id: str,
        *,
        mutation: bool,
    ) -> WikiEvidence | None:
        if (
            self._runtime.session.work_dir_meta.kaos != local_kaos.name
            or self._runtime.wiki is None
            or self._runtime.workspace_id is None
        ):
            return None
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        source = _verified_workspace_source(self._runtime, raw_path)
        if source is None:
            return None
        assert source.path is not None
        return await self._record(
            root_turn_id=root_turn_id,
            tool_call_id=tool_call_id,
            source_class="workspace-mutation" if mutation else "workspace-file",
            request_hash=request_hash,
            result_hash=source.content_hash,
            logical_paths=(source.path,),
            source_refs=(source,),
            reliable=True,
            stable_snapshot=True,
            triggering=True,
        )

    async def _web_document(
        self,
        root_turn_id: str,
        arguments: JsonType,
        request_hash: str,
        result: ToolReturnValue,
        tool_call_id: str,
    ) -> WikiEvidence | None:
        raw_url = _string_argument(arguments, "url")
        normalized_url = _normalize_web_url(raw_url)
        if normalized_url is None:
            return None
        result_hash = _result_hash(result)
        source = SourceRef.model_validate(
            {"kind": "web", "url": normalized_url, "content_hash": result_hash}
        )
        return await self._record(
            root_turn_id=root_turn_id,
            tool_call_id=tool_call_id,
            source_class="web-document",
            request_hash=request_hash,
            result_hash=result_hash,
            source_refs=(source,),
            reliable=True,
            stable_snapshot=True,
            triggering=True,
        )

    async def _record(
        self,
        *,
        root_turn_id: str,
        tool_call_id: str,
        source_class: Literal[
            "workspace-file",
            "workspace-search",
            "shell-result",
            "web-search",
            "web-document",
            "workspace-mutation",
        ],
        request_hash: str,
        result_hash: str,
        logical_paths: tuple[str, ...] = (),
        source_refs: tuple[SourceRef, ...] = (),
        reliable: bool,
        stable_snapshot: bool,
        triggering: bool,
    ) -> WikiEvidence | None:
        return await self._coordinator.record_evidence(
            EvidenceObservation(
                root_turn_id=root_turn_id,
                workspace_id=self._runtime.workspace_id,
                producer_role=self._producer_role,
                producer_id=self._producer_id,
                run_generation=self._run_generation,
                tool_call_id=tool_call_id,
                source_class=source_class,
                request_hash=request_hash,
                result_hash=result_hash,
                logical_paths=logical_paths,
                source_refs=source_refs,
                reliable=reliable,
                stable_snapshot=stable_snapshot,
                triggering=triggering,
            )
        )


def _verified_workspace_source(runtime: Runtime, raw_path: str) -> SourceRef | None:
    workspace = Path(str(runtime.session.work_dir)).expanduser().resolve(strict=True)
    unresolved = Path(raw_path).expanduser()
    if not unresolved.is_absolute():
        if ".." in PurePath(raw_path).parts:
            return None
        unresolved = workspace / unresolved
    try:
        lexical_relative = unresolved.relative_to(workspace)
        candidate = unresolved.resolve(strict=True)
        relative = candidate.relative_to(workspace)
    except (OSError, ValueError):
        return None
    current = workspace
    for component in lexical_relative.parts:
        current = current / component
        if current.is_symlink():
            return None
    if not candidate.is_file() or is_sensitive_file(relative.as_posix()):
        return None
    assert runtime.wiki is not None
    assert runtime.workspace_id is not None
    try:
        return runtime.wiki.registry.relative_source(runtime.workspace_id, candidate)
    except (OSError, ValueError):
        return None


def _arguments_hash(arguments: JsonType) -> str:
    try:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(type(arguments).__name__).encode("utf-8")
    return content_hash(encoded)


def _result_hash(result: ToolReturnValue) -> str:
    return content_hash(_output_bytes(result))


def _normalized_shell_arguments(arguments: JsonType) -> JsonType:
    if not isinstance(arguments, dict):
        return arguments
    normalized = dict(arguments)
    command = normalized.get("command")
    if isinstance(command, str):
        normalized["command"] = " ".join(command.split())
    return cast(JsonType, normalized)


def _normalized_shell_result_hash(result: ToolReturnValue) -> str:
    if isinstance(result.output, str):
        normalized = result.output.replace("\r\n", "\n").replace("\r", "\n")
        return content_hash(normalized.encode("utf-8")[:_MAX_RESULT_BYTES])
    return _result_hash(result)


def _output_bytes(result: ToolReturnValue) -> bytes:
    output = result.output
    if isinstance(output, str):
        encoded = output.encode("utf-8")
    else:
        serializable = [
            part.model_dump(mode="json") if hasattr(part, "model_dump") else str(part)
            for part in output
        ]
        encoded = json.dumps(
            serializable,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return encoded[:_MAX_RESULT_BYTES]


def _output_text(result: ToolReturnValue) -> str:
    if isinstance(result.output, str):
        return result.output
    return "\n".join(
        str(getattr(part, "text", "")) for part in result.output if getattr(part, "text", None)
    )


def _has_non_empty_output(result: ToolReturnValue) -> bool:
    return (
        bool(_output_text(result).strip())
        if not isinstance(result.output, str)
        else bool(result.output.strip())
    )


def _string_argument(arguments: JsonType, name: str) -> str:
    if not isinstance(arguments, dict):
        return ""
    value = arguments.get(name)
    return value if isinstance(value, str) else ""


def _bool_argument(arguments: JsonType, name: str) -> bool:
    if not isinstance(arguments, dict):
        return False
    return arguments.get(name) is True


def _normalize_web_url(raw_url: str) -> str | None:
    if not raw_url or has_url_credentials(raw_url):
        return None
    try:
        parts = urlsplit(raw_url)
        if parts.scheme.casefold() not in {"http", "https"} or not parts.hostname:
            return None
        hostname = parts.hostname.encode("idna").decode("ascii").casefold()
        if ":" in hostname:
            hostname = f"[{hostname}]"
        port = f":{parts.port}" if parts.port is not None else ""
        normalized = urlunsplit(
            (
                parts.scheme.casefold(),
                f"{hostname}{port}",
                parts.path or "/",
                parts.query,
                parts.fragment,
            )
        )
    except (UnicodeError, ValueError):
        return None
    return normalized if not has_url_credentials(normalized) else None


def _is_transient_shell(command: str, output: str) -> bool:
    normalized_command = " ".join(command.split())
    return bool(
        not normalized_command
        or _TRANSIENT_COMMAND.fullmatch(normalized_command)
        or _TEST_COUNT_ONLY.fullmatch(output)
    )


def _is_reusable_conclusion(conclusion: str) -> bool:
    if len(conclusion) < 2:
        return False
    folded = " ".join(conclusion.casefold().split())
    return folded not in {
        "done",
        "task complete",
        "completed successfully",
        "no changes needed",
        "nothing to do",
    }
