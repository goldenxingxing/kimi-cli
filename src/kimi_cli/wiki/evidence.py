"""Typed, runtime-local evidence capture for root Wiki checkpoints."""

from __future__ import annotations

import json
import re
import shlex
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
_MAX_DISCOVERY_MATCHES = 64
_NON_TRANSIENT_CLASSES = frozenset(
    {
        "workspace-file",
        "workspace-search",
        "shell-result",
        "web-search",
        "web-document",
        "workspace-mutation",
    }
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
            verified_search = _verified_workspace_search(
                self._runtime,
                tool_type,
                arguments,
                result,
            )
            if verified_search is None:
                return None
            logical_paths, source_refs = verified_search
            evidence = await self._record(
                root_turn_id=root_turn_id,
                tool_call_id=tool_call_id,
                source_class="workspace-search",
                request_hash=request_hash,
                result_hash=_result_hash(result),
                logical_paths=logical_paths,
                source_refs=source_refs,
                reliable=False,
                stable_snapshot=False,
                triggering=True,
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
                    and not _is_transient_shell(command, output, result)
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
                triggering=True,
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
        evidence_ids = tuple(record.evidence_id for record in triggering[:_MAX_CHECKPOINT_EVIDENCE])
        merged = await self._coordinator.attach_root_evidence_to_equivalent_subagent(
            summary_hash=summary_hash,
            evidence_ids=evidence_ids,
        )
        if merged is not None:
            return merged
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


def _verified_workspace_search(
    runtime: Runtime,
    tool_type: type[object],
    arguments: JsonType,
    result: ToolReturnValue,
) -> tuple[tuple[str, ...], tuple[SourceRef, ...]] | None:
    if (
        runtime.session.work_dir_meta.kaos != local_kaos.name
        or runtime.wiki is None
        or runtime.workspace_id is None
        or not isinstance(arguments, dict)
    ):
        return None
    try:
        workspace = Path(str(runtime.session.work_dir)).expanduser().resolve(strict=True)
    except OSError:
        return None

    if tool_type is Glob:
        raw_target = arguments.get("directory")
        if raw_target is None:
            raw_target = str(workspace)
        if not isinstance(raw_target, str) or not raw_target:
            return None
        search_target = _verified_contained_path(workspace, raw_target)
        if search_target is None or not search_target.is_dir():
            return None
        search_base = search_target
        raw_matches = tuple(line for line in _output_text(result).splitlines() if line)
        require_file = False
    elif tool_type is Grep:
        raw_target = arguments.get("path", ".")
        if not isinstance(raw_target, str) or not raw_target:
            return None
        search_target = _verified_contained_path(workspace, raw_target)
        if search_target is None or not (search_target.is_file() or search_target.is_dir()):
            return None
        search_base = search_target.parent if search_target.is_file() else search_target
        raw_matches = _grep_match_paths(
            _output_text(result),
            arguments.get("output_mode", "files_with_matches"),
        )
        if raw_matches is None:
            return None
        require_file = True
    else:
        return None

    if not raw_matches or len(raw_matches) > _MAX_DISCOVERY_MATCHES:
        return None
    logical_paths: list[str] = []
    source_refs: list[SourceRef] = []
    for raw_match in raw_matches:
        match = _verified_contained_path(workspace, raw_match, relative_to=search_base)
        if match is None or (require_file and not match.is_file()):
            return None
        try:
            relative = match.relative_to(workspace).as_posix()
        except ValueError:
            return None
        if is_sensitive_file(relative):
            return None
        if relative not in logical_paths:
            logical_paths.append(relative)
        if match.is_file():
            try:
                source = runtime.wiki.registry.relative_source(runtime.workspace_id, match)
            except (OSError, ValueError):
                return None
            if all(existing.path != source.path for existing in source_refs):
                source_refs.append(source)
    return tuple(logical_paths), tuple(source_refs)


def _verified_contained_path(
    workspace: Path,
    raw_path: str,
    *,
    relative_to: Path | None = None,
) -> Path | None:
    unresolved = Path(raw_path).expanduser()
    if not unresolved.is_absolute():
        if ".." in PurePath(raw_path).parts:
            return None
        unresolved = (relative_to or workspace) / unresolved
    try:
        lexical_relative = unresolved.relative_to(workspace)
        candidate = unresolved.resolve(strict=True)
        candidate.relative_to(workspace)
    except (OSError, ValueError):
        return None
    current = workspace
    for component in lexical_relative.parts:
        current = current / component
        if current.is_symlink():
            return None
    return candidate


def _grep_match_paths(output: str, output_mode: object) -> tuple[str, ...] | None:
    if output_mode not in {"content", "count_matches", "files_with_matches"}:
        return None
    paths: list[str] = []
    content_line = re.compile(r"^(.*?)([:\-])(\d+)\2")
    for line in output.splitlines():
        if not line:
            continue
        if output_mode == "content":
            if line == "--":
                continue
            matched = content_line.match(line)
            if matched is None:
                return None
            path = matched.group(1)
        elif output_mode == "count_matches":
            path, separator, count = line.rpartition(":")
            if not separator or not count.isdigit():
                return None
        else:
            path = line
        if not path:
            return None
        paths.append(path)
    return tuple(paths)


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


def _is_transient_shell(command: str, output: str, result: ToolReturnValue) -> bool:
    if not command.strip() or _TEST_COUNT_ONLY.fullmatch(output):
        return True
    if any(_transient_brief(block) for block in result.display):
        return True
    pipeline = _shell_pipeline_tokens(command)
    return pipeline is not None and all(_is_transient_command(segment) for segment in pipeline)


def _is_transient_command(tokens: list[str]) -> bool:
    unwrapped = _strip_shell_prefixes(tokens)
    if not unwrapped:
        return False
    executable = unwrapped[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
    arguments = unwrapped[1:]
    if executable == "date":
        return all(_readonly_date_argument(argument) for argument in arguments)
    if executable == "pwd":
        return all(
            argument in {"-L", "-P", "--logical", "--physical", "--help", "--version"}
            for argument in arguments
        )
    if executable in {"ps", "pgrep", "top", "jobs", "uptime"}:
        return True
    if executable == "git":
        return _is_git_status(arguments)
    return False


def _shell_pipeline_tokens(command: str) -> tuple[list[str], ...] | None:
    if "\n" in command or "\r" in command:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return None
    if not tokens or any("`" in token or "$" in token for token in tokens):
        return None
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token == "|":
            if not segments[-1]:
                return None
            segments.append([])
        elif token and set(token) <= set(";&|<>()"):
            return None
        else:
            segments[-1].append(token)
    if not segments[-1]:
        return None
    return tuple(segments)


def _strip_shell_prefixes(tokens: list[str]) -> list[str] | None:
    remaining = list(tokens)
    while remaining:
        token = remaining[0]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
            remaining.pop(0)
            continue
        executable = token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if executable in {"builtin", "exec"}:
            remaining.pop(0)
            if remaining and remaining[0] == "--":
                remaining.pop(0)
            continue
        if executable == "command":
            remaining.pop(0)
            while remaining and remaining[0] in {"-p", "--"}:
                remaining.pop(0)
            continue
        if executable == "time":
            remaining.pop(0)
            if remaining and remaining[0] in {"-p", "--portability"}:
                remaining.pop(0)
            continue
        if executable == "env":
            remaining = _strip_env_prefix(remaining)
        elif executable == "sudo":
            remaining = _strip_sudo_prefix(remaining)
        elif executable == "nice":
            remaining = _strip_nice_prefix(remaining)
        elif executable == "timeout":
            remaining = _strip_timeout_prefix(remaining)
        else:
            break
        if remaining is None:
            return None
    return remaining


def _strip_env_prefix(tokens: list[str]) -> list[str] | None:
    remaining = tokens[1:]
    if remaining and remaining[0] == "--":
        remaining.pop(0)
    while remaining:
        option = remaining[0]
        if option == "--":
            remaining.pop(0)
            break
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", option) or option in {
            "-i",
            "--ignore-environment",
            "-0",
            "--null",
        }:
            remaining.pop(0)
        elif option in {"-u", "--unset", "-C", "--chdir"}:
            if len(remaining) < 2:
                return None
            del remaining[:2]
        elif option.startswith(("--unset=", "--chdir=")):
            remaining.pop(0)
        elif option.startswith("-"):
            return None
        else:
            break
    return remaining


def _strip_sudo_prefix(tokens: list[str]) -> list[str] | None:
    remaining = tokens[1:]
    flags = {
        "-A",
        "-b",
        "-E",
        "-H",
        "-K",
        "-k",
        "-n",
        "-S",
        "--askpass",
        "--background",
        "--non-interactive",
        "--preserve-env",
        "--remove-timestamp",
        "--reset-timestamp",
        "--set-home",
        "--stdin",
    }
    with_value = {
        "-C",
        "-D",
        "-g",
        "-h",
        "-p",
        "-R",
        "-r",
        "-T",
        "-t",
        "-u",
        "--chdir",
        "--close-from",
        "--command-timeout",
        "--group",
        "--host",
        "--prompt",
        "--role",
        "--type",
        "--user",
    }
    while remaining and remaining[0].startswith("-"):
        option = remaining[0]
        if option == "--":
            remaining.pop(0)
            break
        if option in flags or option.startswith("--preserve-env="):
            remaining.pop(0)
        elif option in with_value:
            if len(remaining) < 2:
                return None
            del remaining[:2]
        elif (
            any(option.startswith(f"{name}=") for name in with_value if name.startswith("--"))
            or re.fullmatch(r"-[AbEHKnSk]+", option)
            or re.fullmatch(r"-(?:C|D|g|h|p|R|r|T|t|u).+", option)
        ):
            remaining.pop(0)
        else:
            return None
    return remaining


def _strip_nice_prefix(tokens: list[str]) -> list[str] | None:
    remaining = tokens[1:]
    if remaining and remaining[0] == "--":
        remaining.pop(0)
    elif remaining and remaining[0] in {"-n", "--adjustment"}:
        if len(remaining) < 2 or not re.fullmatch(r"[+-]?\d+", remaining[1]):
            return None
        del remaining[:2]
    elif remaining and (
        re.fullmatch(r"-\d+", remaining[0]) or re.fullmatch(r"--adjustment=[+-]?\d+", remaining[0])
    ):
        remaining.pop(0)
    elif remaining and remaining[0].startswith("-"):
        return None
    return remaining


def _strip_timeout_prefix(tokens: list[str]) -> list[str] | None:
    remaining = tokens[1:]
    flags = {"--foreground", "--preserve-status", "-v", "--verbose"}
    with_value = {"-k", "--kill-after", "-s", "--signal"}
    while remaining and remaining[0].startswith("-"):
        option = remaining[0]
        if option == "--":
            remaining.pop(0)
            break
        if option in flags:
            remaining.pop(0)
        elif option in with_value:
            if len(remaining) < 2:
                return None
            del remaining[:2]
        elif option.startswith(("--kill-after=", "--signal=")):
            remaining.pop(0)
        else:
            return None
    if not remaining or not re.fullmatch(
        r"(?:\d+(?:\.\d+)?(?:[smhd])?|infinity)", remaining[0], re.IGNORECASE
    ):
        return None
    return remaining[1:]


def _readonly_date_argument(argument: str) -> bool:
    return bool(
        argument.startswith("+")
        or argument
        in {
            "-u",
            "--utc",
            "-R",
            "--rfc-email",
            "--resolution",
            "--help",
            "--version",
        }
        or argument == "-I"
        or argument.startswith(("-I", "--iso-8601=", "--rfc-3339="))
    )


def _is_git_status(arguments: list[str]) -> bool:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}:
            index += 2
        elif argument.startswith(("--git-dir=", "--work-tree=", "--namespace=")) or argument in {
            "--no-pager",
            "--literal-pathspecs",
            "--no-optional-locks",
        }:
            index += 1
        else:
            break
    return index < len(arguments) and arguments[index] == "status"


def _transient_brief(block: object) -> bool:
    text = getattr(block, "text", None)
    if not isinstance(text, str):
        return False
    normalized = "-".join(text.casefold().replace("_", "-").split())
    return normalized in {"clock", "current-status", "progress"}


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
