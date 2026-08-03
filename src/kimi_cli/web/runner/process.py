"""Session process management for Kimi CLI web interface."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import mimetypes
import os
import sys
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from kosong.message import ContentPart, ImageURLPart, TextPart
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import TypeAdapter
from starlette.websockets import WebSocket, WebSocketState

from kimi_cli import logger

try:
    from pillow_heif import register_heif_opener  # type: ignore[import-not-found]

    register_heif_opener()
except Exception:  # pragma: no cover - degraded mode if pillow-heif missing
    logger.debug("pillow-heif not available; uploaded HEIC/HEIF will fail to decode")
from kimi_cli.config import LLMModel, load_config
from kimi_cli.llm import ModelCapability, derive_model_capabilities
from kimi_cli.share import get_share_dir
from kimi_cli.utils.subprocess_env import get_clean_env
from kimi_cli.web.models import (
    SessionNoticeEvent,
    SessionNoticePayload,
    SessionState,
    SessionStatus,
)
from kimi_cli.web.runner.messages import new_session_status_message
from kimi_cli.web.store.sessions import load_session_by_id
from kimi_cli.wire.jsonrpc import (
    JSONRPCCancelMessage,
    JSONRPCErrorObject,
    JSONRPCErrorResponse,
    JSONRPCEventMessage,
    JSONRPCInMessage,
    JSONRPCInMessageAdapter,
    JSONRPCOutMessage,
    JSONRPCPromptMessage,
    JSONRPCRequestMessage,
    JSONRPCSuccessResponse,
)
from kimi_cli.wire.serde import deserialize_wire_message

JSONRPCOutMessageAdapter = TypeAdapter[JSONRPCOutMessage](JSONRPCOutMessage)


def worker_log_path() -> Path | None:
    """Path of the log file a crashing worker's traceback ends up in.

    The worker calls ``enable_logging``, which swaps its process-level ``fd=2``
    into loguru. So when it dies there is nothing on the stderr pipe we read
    here, and the whole traceback is in this file instead. Name it in the error
    rather than saying "check the logs", which does not say which ones.

    Returns ``None`` if the share directory cannot be resolved, so that
    reporting one failure never raises a second one.
    """
    try:
        return get_share_dir() / "logs" / "kimi.log"
    except OSError:
        return None


# How far behind one browser tab may fall before it is dropped. A client this
# many messages (or bytes) behind cannot catch up on the live stream anyway;
# reconnecting replays `wire.jsonl` from the start, which is both correct and
# cheaper than holding the backlog.
WS_SEND_QUEUE_MAX_MESSAGES = 8192
WS_SEND_QUEUE_MAX_BYTES = 64 * 1024 * 1024
# A single frame that has not made it onto the wire in this long means the peer
# has stopped reading and is not coming back.
WS_SEND_TIMEOUT_S = 60.0
# Closing a wedged socket writes a close frame, which can block for the same
# reason the data did. Never wait on it.
WS_CLOSE_TIMEOUT_S = 5.0


class _WebSocketChannel:
    """The outbound half of one attached WebSocket.

    Every message the session produces used to be written to each attached
    socket inline, from the task that reads the worker's stdout. A browser tab
    that stopped draining its socket therefore stopped the read loop, the
    worker's stdout pipe filled, and the worker blocked mid-turn — while
    `wire.jsonl` kept being written, so reloading the page showed a turn that
    had in fact finished. Giving each connection its own queue and writer task
    keeps that backpressure where it belongs: on that one connection.
    """

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.dead = False
        self._queue: deque[str] = deque()
        self._queued_bytes = 0
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._on_dead: Callable[[WebSocket], Awaitable[None]] | None = None

    def enqueue(self, message: str) -> bool:
        """Queue a message for this connection. Never blocks.

        Returns False if the client has fallen too far behind to be worth
        keeping, in which case the caller should detach it.
        """
        if self.dead:
            return False
        if (
            len(self._queue) >= WS_SEND_QUEUE_MAX_MESSAGES
            or self._queued_bytes >= WS_SEND_QUEUE_MAX_BYTES
        ):
            logger.warning(
                "WebSocket send queue overflow "
                f"({len(self._queue)} msgs / {self._queued_bytes} bytes); dropping client"
            )
            self.dead = True
            return False
        self._queue.append(message)
        self._queued_bytes += len(message)
        self._wakeup.set()
        return True

    def start_writing(self, on_dead: Callable[[WebSocket], Awaitable[None]]) -> None:
        """Leave replay mode and start draining the queue to the socket."""
        self._on_dead = on_dead
        if self._task is None and not self.dead:
            self._task = asyncio.create_task(self._write_loop())

    async def _write_loop(self) -> None:
        try:
            while True:
                if not self._queue:
                    self._wakeup.clear()
                    if not self._queue:
                        await self._wakeup.wait()
                    continue
                message = self._queue.popleft()
                self._queued_bytes -= len(message)
                if self.ws.client_state != WebSocketState.CONNECTED:
                    break
                try:
                    await asyncio.wait_for(self.ws.send_text(message), WS_SEND_TIMEOUT_S)
                except asyncio.CancelledError:
                    raise
                except TimeoutError:
                    logger.warning(
                        f"websocket send stalled for {WS_SEND_TIMEOUT_S:.0f}s; dropping client"
                    )
                    break
                except Exception as e:
                    logger.warning(f"websocket failed: {e.__class__.__name__} {e}")
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"websocket writer crashed: {e.__class__.__name__} {e}")

        self.dead = True
        self._queue.clear()
        self._queued_bytes = 0
        if self._on_dead is not None:
            await self._on_dead(self.ws)

    async def shutdown(self) -> None:
        """Stop the writer task and discard anything still queued."""
        self.dead = True
        self._on_dead = None
        task = self._task
        self._task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._queue.clear()
        self._queued_bytes = 0


class SessionProcess:
    """Manages a single session's KimiCLI subprocess.

    Handles:
    - Starting/stopping the subprocess
    - Reading from stdout (wire messages from KimiCLI)
    - Writing to stdin (user input to KimiCLI)
    - Broadcasting messages to connected WebSockets

    Concurrency model:
    - `SessionProcess` is the long-lived container for a `session_id`.
      It may outlive worker restarts.
    - Liveness vs busy are separate:
      - `is_alive` / `is_running`: worker subprocess exists and has not exited.
      - `is_busy`: there is at least one in-flight prompt id.
    - WebSocket fanout supports "join while running":
      - New clients replay `wire.jsonl` history first.
      - Live messages during replay are buffered per-WS and flushed afterwards.

    Locks:
    - `_lock` guards worker lifecycle and busy state.
    - `_ws_lock` guards WebSocket state.
    """

    def __init__(self, session_id: UUID) -> None:
        """Initialize a session process."""
        self.session_id = session_id
        self._in_flight_prompt_ids: set[str] = set()
        self._status_seq = 0
        self._worker_id: str | None = None
        self._status = SessionStatus(
            session_id=self.session_id,
            state="stopped",
            seq=self._status_seq,
            worker_id=self._worker_id,
            reason=None,
            detail=None,
            updated_at=datetime.now(UTC),
        )
        self._process: asyncio.subprocess.Process | None = None
        self._channels: dict[WebSocket, _WebSocketChannel] = {}
        self._websocket_count = 0
        self._closing_tasks: set[asyncio.Task[None]] = set()
        self._read_task: asyncio.Task[None] | None = None
        self._expecting_exit = False
        self._lock = asyncio.Lock()
        self._ws_lock = asyncio.Lock()
        self._sent_files: set[str] = set()

    @property
    def is_alive(self) -> bool:
        """Whether the worker subprocess exists and has not exited."""
        process = self._process
        return process is not None and process.returncode is None

    @property
    def is_running(self) -> bool:
        """Backward-compatible name: indicates worker liveness."""
        return self.is_alive

    @property
    def is_busy(self) -> bool:
        """Whether the session is currently processing a prompt."""
        return len(self._in_flight_prompt_ids) > 0

    def clear_in_flight(self) -> None:
        """Clear stale in-flight prompt IDs (e.g. after an error)."""
        self._in_flight_prompt_ids.clear()

    @property
    def status(self) -> SessionStatus:
        """Current runtime status snapshot."""
        return self._status

    @property
    def websocket_count(self) -> int:
        """Get the number of connected WebSockets."""
        return self._websocket_count

    async def send_status_snapshot(self, ws: WebSocket) -> None:
        """Send the current status snapshot to a specific WebSocket.

        Goes through that connection's queue so it cannot overtake the live
        messages flushed when replay ended, and so a stalled peer cannot block
        the caller.
        """
        await self._send_to(ws, new_session_status_message(self._status).model_dump_json())

    def _build_status(
        self,
        state: SessionState,
        reason: str | None,
        detail: str | None,
    ) -> SessionStatus | None:
        """Build a new status object if different from current."""
        current = self._status
        if (
            current.state == state
            and current.reason == reason
            and current.detail == detail
            and current.worker_id == self._worker_id
        ):
            return None
        self._status_seq += 1
        status = SessionStatus(
            session_id=self.session_id,
            state=state,
            seq=self._status_seq,
            worker_id=self._worker_id,
            reason=reason,
            detail=detail,
            updated_at=datetime.now(UTC),
        )
        self._status = status
        return status

    async def _emit_status(
        self,
        state: SessionState,
        *,
        reason: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Emit a status update if different from current."""
        status = self._build_status(state, reason, detail)
        if status is None:
            return
        await self._broadcast(new_session_status_message(status).model_dump_json())

    async def start(
        self,
        *,
        reason: str | None = None,
        detail: str | None = None,
        restart_started_at: float | None = None,
    ) -> None:
        """Start the KimiCLI subprocess."""
        async with self._lock:
            if self.is_alive:
                if self._read_task is None or self._read_task.done():
                    self._read_task = asyncio.create_task(self._read_loop())
                return

            self._in_flight_prompt_ids.clear()
            self._expecting_exit = False
            self._worker_id = str(uuid4())

            # 16MB buffer for large messages (e.g., base64-encoded images)
            STREAM_LIMIT = 16 * 1024 * 1024

            if getattr(sys, "frozen", False):
                worker_cmd = [sys.executable, "__web-worker", str(self.session_id)]
            else:
                worker_cmd = [
                    sys.executable,
                    "-m",
                    "kimi_cli.web.runner.worker",
                    str(self.session_id),
                ]

            self._process = await asyncio.create_subprocess_exec(
                *worker_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=STREAM_LIMIT,
                env=get_clean_env(),
            )

            self._read_task = asyncio.create_task(self._read_loop())
            if restart_started_at is not None:
                elapsed_ms = int((time.perf_counter() - restart_started_at) * 1000)
                detail = f"restart_ms={elapsed_ms}"
                await self._emit_status("idle", reason=reason or "start", detail=detail)
                await self._emit_restart_notice(reason=reason, restart_ms=elapsed_ms)
            else:
                await self._emit_status("idle", reason=reason or "start", detail=None)

    async def stop(self) -> None:
        """Stop the session: terminate worker and close all WebSockets."""
        await self.stop_worker(reason="stop")
        await self._close_all_websockets()

    async def stop_worker(
        self,
        *,
        reason: str | None = None,
        emit_status: bool = True,
    ) -> None:
        """Stop only the worker subprocess, keeping WebSockets connected."""
        async with self._lock:
            self._expecting_exit = True
            if self._process is not None:
                if self._process.returncode is None:
                    self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=10.0)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
                self._process = None

            if self._read_task is not None:
                self._read_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._read_task
                self._read_task = None

            self._in_flight_prompt_ids.clear()
            self._worker_id = None
            self._expecting_exit = False
            if emit_status:
                await self._emit_status("stopped", reason=reason or "stop")

    async def restart_worker(self, *, reason: str | None = None) -> None:
        """Restart the worker subprocess without disconnecting WebSockets."""
        started_at = time.perf_counter()
        await self._emit_status("restarting", reason=reason or "restart")
        await self.stop_worker(reason="restart", emit_status=False)
        await self.start(reason=reason or "restart", restart_started_at=started_at)

    async def _emit_restart_notice(self, *, reason: str | None, restart_ms: int) -> None:
        """Emit a restart notice to all WebSockets."""
        label = "Session restarted"
        if reason == "config_update":
            label = "Session restarted due to config update"
        payload = SessionNoticePayload(
            text=f"{label} · {restart_ms}ms",
            kind="restart",
            reason=reason,
            restart_ms=restart_ms,
        )
        event = SessionNoticeEvent(payload=payload)
        await self._broadcast(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "event",
                    "params": event.model_dump(mode="json"),
                },
                ensure_ascii=False,
            )
        )

    async def _read_loop(self) -> None:
        """Read messages from subprocess stdout and broadcast to WebSockets."""
        assert self._process is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    if self._process.stdout.at_eof():
                        if self._expecting_exit:
                            break
                        stderr = await self._process.stderr.read()
                        # On Windows (ProactorEventLoop), asyncio does not
                        # automatically set returncode when the pipe reaches EOF.
                        # Call wait() explicitly to ensure it is populated.
                        if self._process.returncode is None:
                            await self._process.wait()
                        if not stderr:
                            log_path = worker_log_path()
                            where = (
                                f"Its stderr is redirected into {log_path} —"
                                " the traceback is there."
                                if log_path is not None
                                else "Check logs for details."
                            )
                            err_msg = (
                                "Worker process exited unexpectedly"
                                f" (exit code {self._process.returncode}). "
                                f"{where}"
                            )
                            stderr = err_msg.encode()
                        # Clear in-flight IDs before broadcasting so that
                        # is_busy is already False when the frontend reacts
                        # to the error and sends a new prompt.
                        self._in_flight_prompt_ids.clear()
                        await self._broadcast(
                            JSONRPCErrorResponse(
                                id=str(uuid4()),
                                error=JSONRPCErrorObject(
                                    code=self._process.returncode or -1,
                                    message=stderr.decode("utf-8"),
                                ),
                            ).model_dump_json()
                        )
                        logger.warning(
                            f"Process exited with {self._process.returncode}: "
                            f"{stderr.decode('utf-8')}"
                        )
                        await self._emit_status(
                            "error",
                            reason="process_exit",
                            detail=stderr.decode("utf-8"),
                        )
                        break
                    else:
                        continue

                await self._broadcast(line.decode("utf-8").rstrip("\n"))

                # Handle out message
                try:
                    msg = json.loads(line)
                    match msg.get("method"):
                        case "event":
                            msg["params"] = deserialize_wire_message(msg["params"])
                            await self._handle_out_message(JSONRPCEventMessage.model_validate(msg))
                        case "request":
                            msg["params"] = deserialize_wire_message(msg["params"])
                            await self._handle_out_message(
                                JSONRPCRequestMessage.model_validate(msg)
                            )
                        case _:
                            if msg.get("error"):
                                await self._handle_out_message(
                                    JSONRPCErrorResponse.model_validate(msg)
                                )
                            else:
                                await self._handle_out_message(
                                    JSONRPCSuccessResponse.model_validate(msg)
                                )
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSONRPC out message: {line}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Unexpected error in read loop: {e.__class__.__name__} {e}")
            self._in_flight_prompt_ids.clear()
            await self._emit_status("error", reason="read_loop_error", detail=str(e))

    async def _handle_out_message(self, message: JSONRPCOutMessage) -> None:
        """Handle outbound message from worker."""
        match message:
            case JSONRPCSuccessResponse():
                was_busy = self.is_busy
                if message.id in self._in_flight_prompt_ids:
                    self._in_flight_prompt_ids.remove(message.id)
                if was_busy and not self.is_busy:
                    await self._emit_status("idle", reason="prompt_complete")
            case JSONRPCErrorResponse():
                was_busy = self.is_busy
                if message.id in self._in_flight_prompt_ids:
                    self._in_flight_prompt_ids.remove(message.id)
                if was_busy and not self.is_busy:
                    await self._emit_status("idle", reason="prompt_error")
            case _:
                return

    async def _encode_uploaded_files(self) -> AsyncGenerator[ContentPart]:
        """Encode uploaded files for sending to the model."""
        session = load_session_by_id(self.session_id)
        assert session is not None

        uploads_dir = session.kimi_cli_session.dir / "uploads"
        if not uploads_dir.exists():
            return

        # Load .sent marker left by fork to avoid re-sending inherited files.
        # The marker is kept (not deleted) so it survives process restarts.
        sent_marker = uploads_dir / ".sent"
        if sent_marker.exists():
            try:
                already_sent = json.loads(sent_marker.read_text(encoding="utf-8"))
                self._sent_files.update(already_sent)
            except Exception:
                pass

        all_files = sorted(
            (f for f in uploads_dir.iterdir() if f.name != ".sent"),
            key=lambda x: x.name,
        )
        files = [f for f in all_files if f.name not in self._sent_files]

        if not files:
            return

        # Build file list with paths and mime types
        file_infos: list[tuple[Path, str]] = []
        for file in files:
            mime_type, _ = mimetypes.guess_type(file.name)
            file_infos.append((file, mime_type or "application/octet-stream"))

        # Output file list summary
        file_list_lines = ["<uploaded_files>"]
        for idx, (file, _) in enumerate(file_infos, start=1):
            file_list_lines.append(f"{idx}. {file}")
        file_list_lines.append("</uploaded_files>")
        yield TextPart(text="\n".join(file_list_lines) + "\n\n")

        # Text file extensions
        text_extensions = {
            ".txt",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".css",
            ".js",
            ".ts",
            ".py",
            ".sh",
            ".csv",
            ".log",
            ".rst",
            ".toml",
            ".ini",
        }

        # Check model capabilities of the session's effective model
        # (per-session override, falling back to the global default).
        config = load_config()
        capabilities: set[ModelCapability] = set()
        effective_model_name = session.model or config.default_model
        if effective_model_name and effective_model_name in config.models:
            model_config = config.models[effective_model_name]
            capabilities = derive_model_capabilities(model_config)
        else:
            # Fallback: derive from env var when config file has no model entry
            env_model_name = (
                os.environ.get("KIMI_MODEL_NAME")
                or os.environ.get("OPENAI_MODEL_NAME")
                or os.environ.get("ANTHROPIC_MODEL_NAME")
            )
            if env_model_name:
                capabilities = derive_model_capabilities(
                    LLMModel(provider="", model=env_model_name, max_context_size=100_000)
                )
        is_vision = "image_in" in capabilities
        is_video_in = "video_in" in capabilities

        # Process each file
        for file, mime_type in file_infos:
            file_path = str(file)
            ext = file.suffix.lower()

            if is_vision and mime_type.startswith("image/"):
                try:
                    content = file.read_bytes()
                    with Image.open(io.BytesIO(content)) as img:
                        pil_img: PILImage = img
                        width, height = pil_img.size
                        max_side = max(width, height)
                        if max_side > 4096:
                            scale = 4096 / max_side
                            new_size = (int(width * scale), int(height * scale))
                            pil_img = pil_img.resize(  # pyright: ignore[reportUnknownMemberType]
                                new_size
                            )
                        buffer = io.BytesIO()
                        pil_img.save(buffer, format="PNG")
                        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                        tag = f'<image path="{file_path}" content_type="{mime_type}">'
                        yield TextPart(text=tag)
                        yield ImageURLPart(
                            image_url=ImageURLPart.ImageURL(url=f"data:image/png;base64,{encoded}")
                        )
                        yield TextPart(text="</image>\n\n")
                except Exception:
                    logger.exception("Failed to encode uploaded image %s", file_path)
            elif is_video_in and mime_type.startswith("video/"):
                # For video files, emit a <video> tag for frontend display but don't embed content.
                # The agent will use ReadMediaFile tool to read it, which handles video uploads
                # properly.
                yield TextPart(text=f'<video path="{file_path}" content_type="{mime_type}">')
                yield TextPart(text="</video>\n\n")
            elif ext in text_extensions or mime_type.startswith("text/"):
                try:
                    content = file.read_bytes()
                    text_content = content.decode("utf-8", errors="replace")
                    yield TextPart(text=f'<document path="{file_path}" content_type="{mime_type}">')
                    yield TextPart(text=text_content)
                    yield TextPart(text="</document>\n\n")
                except Exception:
                    # Skip files that fail to decode - don't block the upload
                    pass

        # Mark files as sent and persist to disk so state survives server restarts.
        for file in files:
            self._sent_files.add(file.name)
        sent_marker = uploads_dir / ".sent"
        with contextlib.suppress(Exception):
            sent_marker.write_text(json.dumps(sorted(self._sent_files)), encoding="utf-8")

    async def _handle_in_message(self, message: JSONRPCInMessage) -> str | None:
        """Handle inbound message to worker, encoding uploaded files."""
        match message:
            case JSONRPCPromptMessage():
                user_input: list[ContentPart] = []
                async for part in self._encode_uploaded_files():
                    user_input.append(part)
                # Special marker for file-only uploads
                if isinstance(message.params.user_input, str):
                    if message.params.user_input != "KIMI_FILE_UPLOAD_WITHOUT_MESSAGE":
                        user_input.append(TextPart(text=message.params.user_input))
                else:
                    user_input += message.params.user_input
                return json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "prompt",
                        "id": message.id,
                        "params": {
                            "user_input": [part.model_dump(mode="json") for part in user_input],
                        },
                    },
                    ensure_ascii=False,
                )
            case _:
                return None
        return None

    async def _broadcast(self, message: str) -> None:
        """Hand a message to every attached WebSocket. Never blocks on a client.

        This runs on the task that drains the worker's stdout, so it must
        return promptly no matter how slowly any browser tab is reading.
        Delivery is the job of each connection's own writer task.
        """
        async with self._ws_lock:
            channels = list(self._channels.values())

        for channel in channels:
            if not channel.enqueue(message):
                await self._drop_websocket(channel.ws, reason="send queue overflow")

    async def _send_to(self, ws: WebSocket, message: str) -> None:
        """Queue a message for one connection. Never blocks on the client."""
        async with self._ws_lock:
            channel = self._channels.get(ws)
        if channel is None:
            return
        if not channel.enqueue(message):
            await self._drop_websocket(ws, reason="send queue overflow")

    async def _drop_websocket(self, ws: WebSocket, *, reason: str) -> None:
        """Detach a connection that can no longer keep up, and close it.

        Detaching first is what matters: once the socket is out of
        ``_channels`` it can no longer hold anything up, whatever the close
        handshake does. The client reconnects and replays `wire.jsonl`, so
        nothing it missed is lost.
        """
        async with self._ws_lock:
            channel = self._channels.pop(ws, None)
            self._websocket_count = len(self._channels)
        if channel is None:
            return
        logger.warning(f"Dropping WebSocket ({reason}), remaining={self._websocket_count}")
        self._teardown(channel, code=1011, reason="Client fell behind")

    def _teardown(
        self,
        channel: _WebSocketChannel,
        *,
        code: int,
        reason: str,
    ) -> None:
        """Stop a detached connection's writer and close its socket, in the background.

        Both halves can block for the very reason the connection is being torn
        down: cancelling a writer means waiting for a send that may not be
        going anywhere, and the close handshake writes a frame of its own. The
        caller here may be the read loop, so it waits for neither — the socket
        is already out of ``_channels`` and can no longer hold anything up.
        """

        async def teardown() -> None:
            await channel.shutdown()
            if channel.ws.client_state != WebSocketState.CONNECTED:
                return
            # Already gone, or wedged: either way we are done with it.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    channel.ws.close(code=code, reason=reason), WS_CLOSE_TIMEOUT_S
                )

        task = asyncio.create_task(teardown())
        self._closing_tasks.add(task)
        task.add_done_callback(self._closing_tasks.discard)

    async def add_websocket_and_begin_replay(self, ws: WebSocket) -> None:
        """Atomically attach a WebSocket and enter replay mode for it."""
        async with self._ws_lock:
            if ws not in self._channels:
                self._channels[ws] = _WebSocketChannel(ws)
                self._websocket_count = len(self._channels)
        logger.debug(f"WebSocket added (replay mode), count={self._websocket_count}")

    async def end_replay(self, ws: WebSocket) -> None:
        """Start delivering live messages to a websocket after history replay.

        Anything broadcast during the replay is already queued in order, so the
        writer picks up exactly where the replayed history left off.
        """
        async with self._ws_lock:
            channel = self._channels.get(ws)
        if channel is None:
            return
        channel.start_writing(self._on_channel_dead)

    async def _on_channel_dead(self, ws: WebSocket) -> None:
        """A connection's writer gave up: forget it and close the socket."""
        async with self._ws_lock:
            existing = self._channels.get(ws)
            if existing is None or not existing.dead:
                return
            channel = self._channels.pop(ws, None)
            self._websocket_count = len(self._channels)
        if channel is None:
            return
        logger.debug(f"WebSocket writer ended, remaining={self._websocket_count}")
        self._teardown(channel, code=1011, reason="Send failed")

    async def _close_all_websockets(self) -> None:
        """Close all connected WebSockets."""
        async with self._ws_lock:
            channels = list(self._channels.values())
            self._channels.clear()
            self._websocket_count = 0

        async def close(channel: _WebSocketChannel) -> None:
            await channel.shutdown()
            if channel.ws.client_state != WebSocketState.CONNECTED:
                return
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    channel.ws.close(code=1001, reason="Session process exited"),
                    WS_CLOSE_TIMEOUT_S,
                )

        # Concurrently: one wedged peer must not make everyone else wait out
        # its close timeout too.
        await asyncio.gather(*(close(channel) for channel in channels))

    async def remove_websocket(self, ws: WebSocket) -> None:
        """Remove a WebSocket connection from this session."""
        async with self._ws_lock:
            channel = self._channels.pop(ws, None)
            self._websocket_count = len(self._channels)
        if channel is not None:
            logger.debug(f"WebSocket removed, count={self._websocket_count}")
            await channel.shutdown()

    async def apply_compaction_ratio(self, ratio: float) -> bool:
        """Push a new auto-compaction ratio into a live worker.

        Returns whether the worker was running and the message was written.
        A stopped worker needs nothing: it reads the persisted value from
        config.toml the next time it starts.
        """
        if not self.is_running:
            return False
        process = self._process
        if process is None or process.stdin is None:
            return False
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "set_compaction_ratio",
                "id": uuid4().hex,
                "params": {"ratio": ratio},
            },
            ensure_ascii=False,
        )
        try:
            process.stdin.write((payload + "\n").encode())
            await process.stdin.drain()
        except (OSError, RuntimeError):
            logger.warning(
                "Could not push the compaction ratio to session {sid}",
                sid=self.session_id,
            )
            return False
        return True

    async def send_message(self, message: str) -> None:
        """Send a message to the subprocess stdin."""
        await self.start()
        process = self._process
        assert process is not None
        assert process.stdin is not None

        # Handle in message
        try:
            in_message = JSONRPCInMessageAdapter.validate_json(message)
            if isinstance(in_message, JSONRPCPromptMessage):
                was_busy = self.is_busy
                self._in_flight_prompt_ids.add(in_message.id)
                if not was_busy:
                    await self._emit_status("busy", reason="prompt")
            elif isinstance(in_message, JSONRPCCancelMessage) and not self.is_busy:
                # If not busy, return success to avoid errors
                await self._broadcast(
                    JSONRPCSuccessResponse(id=in_message.id, result={}).model_dump_json()
                )
                return

            new_message = await self._handle_in_message(in_message)
            if new_message is not None:
                message = new_message
        except ValueError as e:
            logger.error(f"{e.__class__.__name__} {e}: Invalid JSONRPC in message: {message}")
            return

        process.stdin.write((message + "\n").encode("utf-8"))
        await process.stdin.drain()


class KimiCLIRunner:
    """Manages multiple session processes."""

    def __init__(self) -> None:
        """Initialize the runner."""
        self._sessions: dict[UUID, SessionProcess] = {}
        self._lock = asyncio.Lock()

    def start(self) -> None:
        """Start the runner (no-op, sessions started on demand)."""
        pass

    async def stop(self) -> None:
        """Stop all running sessions."""
        tasks: list[asyncio.Task[None]] = []
        for session in self._sessions.values():
            if session.is_running:
                tasks.append(asyncio.create_task(session.stop()))
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=5.0)
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t

    async def get_or_create_session(self, session_id: UUID) -> SessionProcess:
        """Get or create a session process."""
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionProcess(session_id)
            return self._sessions[session_id]

    def get_session(self, session_id: UUID) -> SessionProcess | None:
        """Get a session process if it exists."""
        return self._sessions.get(session_id)

    async def detach_websocket(self, ws: WebSocket, session_id: UUID) -> None:
        """Detach a WebSocket from a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                await session.remove_websocket(ws)

    async def apply_compaction_ratio(self, ratio: float) -> list[UUID]:
        """Retune every live worker in place; no restart, no interruption."""
        async with self._lock:
            running = [(sid, proc) for sid, proc in self._sessions.items() if proc.is_running]
        applied: list[UUID] = []
        for session_id, proc in running:
            if await proc.apply_compaction_ratio(ratio):
                applied.append(session_id)
        return applied

    async def restart_running_workers(
        self,
        *,
        reason: str,
        force: bool,
        skip_model_override: bool = False,
    ) -> RestartWorkersSummary:
        """Restart all running workers to apply global config updates.

        Args:
            reason: Reason for the restart (e.g., "config_update")
            force: If True, also restart busy sessions (may interrupt prompts)
            skip_model_override: If True, sessions that have a per-session
                model override are not restarted (their model does not depend
                on the global default model).

        Returns:
            Summary of restarted and skipped sessions
        """
        async with self._lock:
            running = [(sid, proc) for sid, proc in self._sessions.items() if proc.is_running]

        restarted: list[UUID] = []
        skipped_busy: list[UUID] = []
        tasks: list[asyncio.Task[None]] = []

        for session_id, proc in running:
            if skip_model_override:
                joint = load_session_by_id(session_id)
                if joint is not None and joint.model:
                    continue
            if proc.is_busy and not force:
                skipped_busy.append(session_id)
                continue
            restarted.append(session_id)
            tasks.append(asyncio.create_task(proc.restart_worker(reason=reason)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return RestartWorkersSummary(
            restarted_session_ids=restarted,
            skipped_busy_session_ids=skipped_busy,
        )


@dataclass(slots=True)
class RestartWorkersSummary:
    """Summary of a restart_running_workers operation."""

    restarted_session_ids: list[UUID]
    skipped_busy_session_ids: list[UUID]
