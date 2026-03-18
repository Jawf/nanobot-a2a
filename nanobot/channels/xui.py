"""XUI channel: A2A protocol (JSON-RPC + SSE) for XUI frontend integration.

Implements the Agent-to-Agent (A2A) protocol endpoints that the XUI console
expects, using only Python stdlib (asyncio) — no extra dependencies required.

Endpoints
---------
GET  /.well-known/agent-card.json  — Agent card discovery
POST /                             — JSON-RPC 2.0 (message/send, message/stream, tasks/get, tasks/cancel)
GET  /serverstatus                 — Health / liveness check
POST /resume                       — Resume interrupted task (stub)
POST /edit_message                 — Edit & re-execute message (stub)
OPTIONS *                          — CORS preflight
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any
from urllib.parse import parse_qs, urlparse

from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class XuiConfig(Base):
    """XUI channel configuration."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 18791
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    response_timeout: int = 120  # seconds to wait for agent response
    agent_name: str = "nanobot"
    agent_description: str = "nanobot AI assistant"
    agent_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Task bookkeeping
# ---------------------------------------------------------------------------


class _TaskInfo:
    """Track a single A2A task lifecycle."""

    __slots__ = ("task_id", "context_id", "state", "history", "created", "updated",
                 "event", "chunks", "done", "sse_queues")

    def __init__(self, task_id: str, context_id: str, user_message: dict) -> None:
        self.task_id = task_id
        self.context_id = context_id
        self.state = "submitted"
        self.history: list[dict] = [user_message]
        self.created = time.time()
        self.updated = time.time()
        # For sync reply
        self.event = asyncio.Event()
        self.chunks: list[str] = []
        self.done = False
        # For SSE streaming
        self.sse_queues: list[asyncio.Queue] = []

    def append_progress(self, text: str) -> None:
        self.chunks.append(text)
        self.state = "working"
        self.updated = time.time()
        # status-update with embedded message (matches agentickit pattern)
        msg = _make_message("agent", text, self.task_id, self.context_id)
        for q in self.sse_queues:
            q.put_nowait({"state": "working", "final": False, "message": msg, "metadata": None})

    def append_data_progress(self, biz_data: dict) -> None:
        """Send a structured data message (e.g. PPT card) as a progress event."""
        self.state = "working"
        self.updated = time.time()
        msg = {
            "kind": "message",
            "messageId": str(uuid.uuid4()),
            "role": "agent",
            "parts": [{"kind": "data", "data": biz_data, "metadata": {}}],
            "taskId": self.task_id,
            "contextId": self.context_id,
            "metadata": {"version": 3, "append": False, "lastChunk": True},
        }
        for q in self.sse_queues:
            q.put_nowait({"state": "working", "final": False, "message": msg, "metadata": None})

    def finish(self, text: str) -> None:
        self.chunks.append(text)
        self.done = True
        self.state = "completed"
        self.updated = time.time()
        agent_msg = _make_message("agent", text, self.task_id, self.context_id)
        self.history.append(agent_msg)
        # 1) status-update with text message (state=working, final=false)
        # 2) status-update final signal (state=working, final=true, empty text, kit_is_final)
        final_msg = _make_message("agent", "", self.task_id, self.context_id, is_final=True)
        for q in self.sse_queues:
            q.put_nowait({"state": "working", "final": False, "message": agent_msg, "metadata": None})
            q.put_nowait({"state": "working", "final": True, "message": final_msg, "metadata": {"kit_is_final": True}})
        self.event.set()

    def fail(self, text: str) -> None:
        """Mark task as failed and send error events to SSE subscribers."""
        self.chunks.append(text)
        self.done = True
        self.state = "failed"
        self.updated = time.time()
        agent_msg = _make_message("agent", text, self.task_id, self.context_id)
        self.history.append(agent_msg)
        final_msg = _make_message("agent", "", self.task_id, self.context_id, is_final=True)
        for q in self.sse_queues:
            q.put_nowait({"state": "failed", "final": False, "message": agent_msg, "metadata": None})
            q.put_nowait({"state": "failed", "final": True, "message": final_msg, "metadata": {"kit_is_final": True}})
        self.event.set()

    @property
    def full_text(self) -> str:
        return "".join(self.chunks)

    def to_task_dict(self) -> dict:
        """Build A2A Task object."""
        status: dict[str, Any] = {
            "state": self.state,
            "timestamp": _iso_now(),
        }
        if self.done and self.full_text:
            status["message"] = _make_message("agent", self.full_text, self.task_id, self.context_id)
        return {
            "kind": "task",
            "id": self.task_id,
            "contextId": self.context_id,
            "status": status,
            "history": self.history,
            "artifacts": [],
            "metadata": {},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB

# Patterns that indicate the agent returned an error rather than a real answer.
_ERROR_PREFIXES = (
    "Error: Error code:",
    "Sorry, I encountered an error",
    "I reached the maximum number of tool call iterations",
)

import re

_RE_HTTP_CODE = re.compile(r"Error code:\s*(\d{3})")
_RE_INNER_MESSAGE = re.compile(r'"message"\s*:\s*"([^"]{10,})"')


def _is_error_content(content: str) -> bool:
    """Return True if *content* looks like an agent error message."""
    return any(content.startswith(p) for p in _ERROR_PREFIXES)


def _extract_error_msg_value(content: str) -> str | None:
    """Extract the error_msg value from a provider error string.

    Handles nested JSON like: 'error_msg': '厂商返回异常：[{ "error": {...} }]'
    The previous regex approach broke on nested braces/quotes; instead we find
    the key and then greedily capture everything up to the outermost closing
    delimiter.
    """
    # Find error_msg key
    idx = content.find("error_msg")
    if idx == -1:
        return None
    # Skip past  error_msg'?: ?'
    rest = content[idx + len("error_msg"):]
    # Skip separator chars:  ' " : =  and whitespace
    i = 0
    while i < len(rest) and rest[i] in ("'", '"', ':', '=', ' ', '\t'):
        i += 1
    rest = rest[i:]
    if not rest:
        return None
    # Determine the closing delimiter — match the quote that opened it, or end
    # of string.  The value may be wrapped in ' or " or bare.
    if rest[0] in ("'", '"'):
        quote = rest[0]
        rest = rest[1:]
        # Find the *last* occurrence of the closing quote followed by optional
        # punctuation — avoids stopping at nested quotes inside JSON.
        end = rest.rfind(quote)
        return rest[:end].strip() if end > 0 else rest.strip()
    # No quote wrapper — take the rest (until end of string)
    return rest.strip().rstrip("}")


def _clean_error_for_display(content: str) -> str:
    """Extract a user-friendly message from a raw LLM/provider error string."""
    # Fallback generic errors pass through as-is
    if content.startswith("Sorry,") or content.startswith("I reached"):
        return content

    # Try to extract HTTP status code
    code_match = _RE_HTTP_CODE.search(content)
    http_code = code_match.group(1) if code_match else "unknown"

    # Try to extract the provider error_msg value (often Chinese + nested JSON)
    raw_msg = _extract_error_msg_value(content)
    if raw_msg:
        # Try to find a clean inner "message" from nested JSON
        inner = _RE_INNER_MESSAGE.search(raw_msg)
        if inner:
            detail = inner.group(1).strip()
            if len(detail) > 500:
                detail = detail[:500] + "…"
            return f"AI 模型调用失败（{http_code}）：{detail}"
        # Use the raw_msg directly
        if len(raw_msg) > 600:
            raw_msg = raw_msg[:600] + "…"
        return f"AI 模型调用失败（{http_code}）：{raw_msg}"

    # Last resort: just show the code and a generic hint
    return f"AI 模型调用失败（HTTP {http_code}），请尝试简化问题或开始新对话。"


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _make_message(role: str, text: str, task_id: str, context_id: str, is_final: bool = False) -> dict:
    """Build an A2A Message dict matching agentickit wire format.

    Message metadata always includes ``version``, ``append``, ``lastChunk``.
    """
    metadata: dict[str, Any] = {"version": 3, "append": False, "lastChunk": True}
    return {
        "kind": "message",
        "messageId": str(uuid.uuid4()),
        "role": role,
        "parts": [{"kind": "text", "text": text}],
        "taskId": task_id,
        "contextId": context_id,
        "metadata": metadata,
    }


def _jsonrpc_success(req_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# ---------------------------------------------------------------------------
# Minimal async HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


class _HttpRequest:
    """Parsed HTTP request."""

    __slots__ = ("method", "path", "headers", "body", "query")

    def __init__(
        self, method: str, path: str, headers: dict[str, str], body: bytes, query: dict[str, str],
    ):
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body
        self.query = query

    def json(self) -> dict:
        return json.loads(self.body) if self.body else {}

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


async def _read_request(reader: asyncio.StreamReader) -> _HttpRequest | None:
    """Read and parse one HTTP/1.1 request from *reader*."""
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=30)
    except (asyncio.TimeoutError, ConnectionError):
        return None
    if not request_line:
        return None

    try:
        parts = request_line.decode("utf-8", errors="replace").strip().split(" ", 2)
        if len(parts) < 2:
            return None
        method, raw_path = parts[0].upper(), parts[1]
    except Exception:
        return None

    parsed = urlparse(raw_path)
    path = parsed.path
    query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        decoded = line.decode("utf-8", errors="replace").strip()
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    body = b""
    content_length = int(headers.get("content-length", "0"))
    if content_length > 0:
        content_length = min(content_length, _MAX_REQUEST_SIZE)
        body = await reader.readexactly(content_length)

    return _HttpRequest(method=method, path=path, headers=headers, body=body, query=query)


# ---------------------------------------------------------------------------
# HTTP response builders
# ---------------------------------------------------------------------------


def _cors_headers(origin: str, allowed: list[str]) -> str:
    if "*" in allowed or origin in allowed:
        allow_origin = origin or "*"
    else:
        allow_origin = allowed[0] if allowed else ""
    return (
        f"Access-Control-Allow-Origin: {allow_origin}\r\n"
        f"Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        f"Access-Control-Allow-Headers: Content-Type, Authorization, X-Session-Id, X-Sender-Id\r\n"
        f"Access-Control-Allow-Credentials: true\r\n"
        f"Access-Control-Max-Age: 3600\r\n"
    )


def _json_response_cors(data: dict | list, status: int, origin: str, allowed: list[str]) -> bytes:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    status_text = {
        200: "OK", 204: "No Content", 400: "Bad Request",
        403: "Forbidden", 404: "Not Found", 500: "Internal Server Error",
        501: "Not Implemented", 504: "Gateway Timeout",
    }.get(status, "Error")
    return (
        f"HTTP/1.1 {status} {status_text}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"{_cors_headers(origin, allowed)}"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("utf-8") + body


def _options_response(origin: str, allowed: list[str]) -> bytes:
    return (
        f"HTTP/1.1 204 No Content\r\n"
        f"{_cors_headers(origin, allowed)}"
        f"Content-Length: 0\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("utf-8")


def _sse_start_headers(origin: str, allowed: list[str]) -> bytes:
    return (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: text/event-stream; charset=utf-8\r\n"
        f"Cache-Control: no-cache\r\n"
        f"Connection: keep-alive\r\n"
        f"X-Accel-Buffering: no\r\n"
        f"{_cors_headers(origin, allowed)}"
        f"\r\n"
    ).encode("utf-8")


def _sse_data(data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


class XuiChannel(BaseChannel):
    """A2A protocol channel for XUI frontend integration.

    Implements the Agent-to-Agent protocol over JSON-RPC 2.0 so that the
    XUI console can discover the agent, send messages, and receive streaming
    responses via Server-Sent Events.
    """

    name = "xui"
    display_name = "XUI"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return XuiConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = XuiConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: XuiConfig = config
        self._server: asyncio.Server | None = None
        # chat_id -> task info
        self._tasks: dict[str, _TaskInfo] = {}
        # task_id -> chat_id (reverse lookup)
        self._task_to_chat: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Agent card
    # ------------------------------------------------------------------

    def _agent_card(self) -> dict:
        """Build the A2A agent card."""
        base_url = f"http://{self.config.host}:{self.config.port}"
        if self.config.host == "0.0.0.0":
            base_url = f"http://127.0.0.1:{self.config.port}"
        return {
            "name": self.config.agent_name,
            "description": self.config.agent_description,
            "url": base_url,
            "version": self.config.agent_version,
            "protocolVersion": "0.2.1",
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            },
            "skills": [],
        }

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        import socket

        self._running = True
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.config.host,
            self.config.port,
            reuse_address=True,
        )
        # On Windows, also set SO_REUSEADDR on each listening socket
        for sock in self._server.sockets or []:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        logger.info(
            "XUI channel (A2A) listening on http://{}:{}",
            self.config.host,
            self.config.port,
        )
        try:
            await self._server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            self._server.close()
            await self._server.wait_closed()

    async def stop(self) -> None:
        self._running = False
        if self._server and self._server.is_serving():
            self._server.close()
            await self._server.wait_closed()
        logger.info("XUI channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Receive an agent reply and route to the appropriate task."""
        chat_id = msg.chat_id
        is_progress = msg.metadata.get("_progress", False)
        is_data_message = msg.metadata.get("_data_message", False)

        logger.info(
            "XUI send(): chat_id={}, is_progress={}, data_msg={}, tasks={}, content={}",
            chat_id, is_progress, is_data_message, list(self._tasks.keys()),
            (msg.content[:80] if msg.content else "(empty)") if not is_data_message else "(data)",
        )

        task = self._tasks.get(chat_id)
        if not task:
            logger.warning("XUI: no task found for chat_id={}, available={}", chat_id, list(self._tasks.keys()))
            return

        if is_data_message:
            biz_data = msg.metadata.get("_biz_data", {})
            task.append_data_progress(biz_data)
            logger.info("XUI: data message delivered to task {}", task.task_id)
        elif is_progress:
            task.append_progress(msg.content)
            logger.debug("XUI: progress chunk delivered to task {}", task.task_id)
        elif _is_error_content(msg.content):
            clean = _clean_error_for_display(msg.content)
            task.fail(clean)
            logger.warning("XUI: error response delivered to task {} — {}", task.task_id, clean)
        else:
            task.finish(msg.content)
            logger.info("XUI: final response delivered to task {}", task.task_id)

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        try:
            req = await _read_request(reader)
            if req is None:
                writer.close()
                return

            origin = req.header("origin", "*")
            allowed = self.config.cors_origins

            # CORS preflight
            if req.method == "OPTIONS":
                writer.write(_options_response(origin, allowed))
                await writer.drain()
                writer.close()
                return

            # Route
            if req.method == "GET" and req.path == "/.well-known/agent-card.json":
                await self._route_agent_card(writer, origin, allowed)

            elif req.method == "GET" and req.path == "/.well-known/agent.json":
                # Deprecated alias
                await self._route_agent_card(writer, origin, allowed)

            elif req.method == "GET" and req.path == "/serverstatus":
                await self._route_serverstatus(writer, origin, allowed)

            elif req.method == "POST" and req.path == "/":
                await self._route_jsonrpc(req, writer, origin, allowed)
                return  # writer managed inside

            elif req.method == "POST" and req.path == "/resume":
                await self._route_stub(req, writer, origin, allowed, "resume not supported")
                return

            elif req.method == "POST" and req.path == "/edit_message":
                await self._route_stub(req, writer, origin, allowed, "edit_message not supported")
                return

            else:
                writer.write(_json_response_cors(
                    {"error": "not found"}, 404, origin, allowed,
                ))
                await writer.drain()

        except (ConnectionResetError, BrokenPipeError) as exc:
            logger.warning("XUI: client disconnected during request: {}", exc)
        except Exception:
            logger.exception("XUI: error handling request")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # GET /.well-known/agent-card.json
    # ------------------------------------------------------------------

    async def _route_agent_card(
        self, writer: asyncio.StreamWriter, origin: str, allowed: list[str],
    ) -> None:
        writer.write(_json_response_cors(self._agent_card(), 200, origin, allowed))
        await writer.drain()

    # ------------------------------------------------------------------
    # GET /serverstatus
    # ------------------------------------------------------------------

    async def _route_serverstatus(
        self, writer: asyncio.StreamWriter, origin: str, allowed: list[str],
    ) -> None:
        writer.write(_json_response_cors(
            {"status": "running", "channel": "xui", "running": self._running},
            200, origin, allowed,
        ))
        await writer.drain()

    # ------------------------------------------------------------------
    # POST / — JSON-RPC 2.0 dispatcher
    # ------------------------------------------------------------------

    async def _route_jsonrpc(
        self,
        req: _HttpRequest,
        writer: asyncio.StreamWriter,
        origin: str,
        allowed: list[str],
    ) -> None:
        """Parse JSON-RPC request and dispatch to the appropriate handler."""
        try:
            body = req.json()
        except (json.JSONDecodeError, Exception):
            writer.write(_json_response_cors(
                _jsonrpc_error(None, -32700, "Parse error"),
                400, origin, allowed,
            ))
            await writer.drain()
            writer.close()
            return

        req_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})
        jsonrpc_version = body.get("jsonrpc", "")

        logger.info("XUI JSON-RPC: method={}, id={}", method, req_id)

        if jsonrpc_version != "2.0":
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32600, "Invalid Request: jsonrpc must be '2.0'"),
                400, origin, allowed,
            ))
            await writer.drain()
            writer.close()
            return

        # Dispatch by method
        if method == "message/send":
            await self._handle_message_send(req_id, params, writer, origin, allowed)
        elif method == "message/stream":
            await self._handle_message_stream(req_id, params, writer, origin, allowed)
        elif method == "tasks/get":
            await self._handle_tasks_get(req_id, params, writer, origin, allowed)
        elif method == "tasks/cancel":
            await self._handle_tasks_cancel(req_id, params, writer, origin, allowed)
        elif method == "tasks/resubscribe":
            await self._handle_tasks_resubscribe(req_id, params, writer, origin, allowed)
        elif method == "agent/authenticatedExtendedCard":
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32007, "Authenticated extended card not configured"),
                200, origin, allowed,
            ))
            await writer.drain()
            writer.close()
        else:
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32601, f"Method not found: {method}"),
                200, origin, allowed,
            ))
            await writer.drain()
            writer.close()

    # ------------------------------------------------------------------
    # message/send — synchronous message exchange
    # ------------------------------------------------------------------

    async def _handle_message_send(
        self,
        req_id: Any,
        params: dict,
        writer: asyncio.StreamWriter,
        origin: str,
        allowed: list[str],
    ) -> None:
        """Handle message/send: accept message, wait for full reply, return Task."""
        message = params.get("message", {})
        text = self._extract_text(message)
        logger.info("XUI message/send: params keys={}, message keys={}", list(params.keys()), list(message.keys()))
        if not text:
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32602, "Invalid params: message with text part required"),
                200, origin, allowed,
            ))
            await writer.drain()
            writer.close()
            return

        task_id = message.get("taskId") or message.get("task_id") or str(uuid.uuid4())
        context_id = message.get("contextId") or message.get("context_id") or str(uuid.uuid4())
        sender_id = message.get("metadata", {}).get("sender_id", "xui-user")
        chat_id = f"xui_{context_id}"
        logger.info("XUI message/send: task_id={}, context_id={}, chat_id={}, text={}",
                     task_id, context_id, chat_id, text[:60])

        if not self.is_allowed(sender_id):
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32600, "Access denied"),
                200, origin, allowed,
            ))
            await writer.drain()
            writer.close()
            return

        # Create task
        user_msg = _make_message("user", text, task_id, context_id)
        task_info = _TaskInfo(task_id, context_id, user_msg)
        self._tasks[chat_id] = task_info
        self._task_to_chat[task_id] = chat_id

        try:
            # Submit to nanobot agent
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=text,
                metadata={"xui": {"task_id": task_id, "context_id": context_id}},
                session_key=f"xui:{context_id}",
            )

            # Wait for agent reply
            try:
                await asyncio.wait_for(task_info.event.wait(), timeout=self.config.response_timeout)
            except asyncio.TimeoutError:
                task_info.state = "failed"
                writer.write(_json_response_cors(
                    _jsonrpc_error(req_id, -32603, "Response timeout", {"partial": task_info.full_text}),
                    200, origin, allowed,
                ))
                await writer.drain()
                writer.close()
                return

            # Return completed task
            resp = _json_response_cors(
                _jsonrpc_success(req_id, task_info.to_task_dict()),
                200, origin, allowed,
            )
            writer.write(resp)
            await writer.drain()
            logger.info("XUI message/send: response written ({} bytes) for task {}", len(resp), task_id)
        except (ConnectionResetError, BrokenPipeError) as exc:
            logger.warning("XUI message/send: client disconnected before response: {}", exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # message/stream — SSE streaming
    # ------------------------------------------------------------------

    async def _handle_message_stream(
        self,
        req_id: Any,
        params: dict,
        writer: asyncio.StreamWriter,
        origin: str,
        allowed: list[str],
    ) -> None:
        """Handle message/stream: accept message, stream SSE events."""
        message = params.get("message", {})
        text = self._extract_text(message)
        logger.info("XUI message/stream: params keys={}, message keys={}", list(params.keys()), list(message.keys()))
        if not text:
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32602, "Invalid params: message with text part required"),
                200, origin, allowed,
            ))
            await writer.drain()
            writer.close()
            return

        task_id = message.get("taskId") or message.get("task_id") or str(uuid.uuid4())
        context_id = message.get("contextId") or message.get("context_id") or str(uuid.uuid4())
        sender_id = message.get("metadata", {}).get("sender_id", "xui-user")
        chat_id = f"xui_{context_id}"
        logger.info("XUI message/stream: task_id={}, context_id={}, chat_id={}, text={}",
                     task_id, context_id, chat_id, text[:60])

        if not self.is_allowed(sender_id):
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32600, "Access denied"),
                200, origin, allowed,
            ))
            await writer.drain()
            writer.close()
            return

        # Create task
        user_msg = _make_message("user", text, task_id, context_id)
        task_info = _TaskInfo(task_id, context_id, user_msg)
        self._tasks[chat_id] = task_info
        self._task_to_chat[task_id] = chat_id

        # SSE queue for this stream
        queue: asyncio.Queue = asyncio.Queue()
        task_info.sse_queues.append(queue)

        try:
            # Start SSE response
            writer.write(_sse_start_headers(origin, allowed))
            await writer.drain()
            logger.info("XUI SSE: headers sent for task {}", task_id)

            # Send initial Task event
            task_info.state = "submitted"
            writer.write(_sse_data(
                _jsonrpc_success(req_id, task_info.to_task_dict())
            ))
            await writer.drain()
            logger.info("XUI SSE: initial task event sent for task {}", task_id)

            # Submit to nanobot agent
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=text,
                metadata={"xui": {"task_id": task_id, "context_id": context_id, "stream": True}},
                session_key=f"xui:{context_id}",
            )
            logger.info("XUI SSE: message submitted to bus, waiting for events...")

            # Stream events from queue
            # Each event dict: {"state": str, "final": bool, "message": dict|None, "metadata": dict|None}
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=self.config.response_timeout)
                except asyncio.TimeoutError:
                    logger.warning("XUI SSE: timeout waiting for events, task {}", task_id)
                    writer.write(_sse_data(
                        _jsonrpc_success(req_id, {
                            "kind": "status-update",
                            "taskId": task_id,
                            "contextId": context_id,
                            "final": True,
                            "status": {"state": "failed", "timestamp": _iso_now(),
                                       "message": _make_message("agent", "Response timeout", task_id, context_id)},
                            "metadata": {},
                        })
                    ))
                    await writer.drain()
                    break

                is_final = event.get("final", False)
                state = event.get("state", "working")
                msg_obj = event.get("message")
                evt_meta = event.get("metadata")

                status_update: dict[str, Any] = {
                    "kind": "status-update",
                    "taskId": task_id,
                    "contextId": context_id,
                    "final": is_final,
                    "status": {
                        "state": state,
                        "timestamp": _iso_now(),
                    },
                }
                if msg_obj is not None:
                    status_update["status"]["message"] = msg_obj
                if evt_meta is not None:
                    status_update["metadata"] = evt_meta

                sse_bytes = _sse_data(_jsonrpc_success(req_id, status_update))
                writer.write(sse_bytes)
                await writer.drain()
                logger.info("XUI SSE: status-update sent state={} final={} ({} bytes) task={}",
                            state, is_final, len(sse_bytes), task_id)

                if is_final:
                    break

        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError) as exc:
            logger.warning("XUI SSE: client disconnected: {} (task {})", exc, task_id)
        finally:
            if queue in task_info.sse_queues:
                task_info.sse_queues.remove(queue)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # tasks/get — get task status
    # ------------------------------------------------------------------

    async def _handle_tasks_get(
        self,
        req_id: Any,
        params: dict,
        writer: asyncio.StreamWriter,
        origin: str,
        allowed: list[str],
    ) -> None:
        task_id = params.get("id", "")
        chat_id = self._task_to_chat.get(task_id)
        task_info = self._tasks.get(chat_id) if chat_id else None

        if not task_info:
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32001, f"Task not found: {task_id}"),
                200, origin, allowed,
            ))
        else:
            writer.write(_json_response_cors(
                _jsonrpc_success(req_id, task_info.to_task_dict()),
                200, origin, allowed,
            ))
        await writer.drain()
        writer.close()

    # ------------------------------------------------------------------
    # tasks/cancel — cancel task
    # ------------------------------------------------------------------

    async def _handle_tasks_cancel(
        self,
        req_id: Any,
        params: dict,
        writer: asyncio.StreamWriter,
        origin: str,
        allowed: list[str],
    ) -> None:
        task_id = params.get("id", "")
        chat_id = self._task_to_chat.get(task_id)
        task_info = self._tasks.get(chat_id) if chat_id else None

        if not task_info:
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32001, f"Task not found: {task_id}"),
                200, origin, allowed,
            ))
        elif task_info.done:
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32002, "Task already completed"),
                200, origin, allowed,
            ))
        else:
            task_info.state = "canceled"
            task_info.done = True
            task_info.event.set()
            # Signal SSE queues
            for q in task_info.sse_queues:
                q.put_nowait({"state": "canceled", "final": True, "message": None, "metadata": None})
            writer.write(_json_response_cors(
                _jsonrpc_success(req_id, task_info.to_task_dict()),
                200, origin, allowed,
            ))
        await writer.drain()
        writer.close()

    # ------------------------------------------------------------------
    # tasks/resubscribe — SSE re-subscription
    # ------------------------------------------------------------------

    async def _handle_tasks_resubscribe(
        self,
        req_id: Any,
        params: dict,
        writer: asyncio.StreamWriter,
        origin: str,
        allowed: list[str],
    ) -> None:
        """Re-subscribe to task events via SSE."""
        task_id = params.get("id", "")
        chat_id = self._task_to_chat.get(task_id)
        task_info = self._tasks.get(chat_id) if chat_id else None

        if not task_info:
            writer.write(_json_response_cors(
                _jsonrpc_error(req_id, -32001, f"Task not found: {task_id}"),
                200, origin, allowed,
            ))
            await writer.drain()
            writer.close()
            return

        # If task already done, return final state
        if task_info.done:
            writer.write(_sse_start_headers(origin, allowed))
            await writer.drain()
            writer.write(_sse_data(_jsonrpc_success(req_id, {
                "kind": "status-update",
                "taskId": task_id,
                "contextId": task_info.context_id,
                "final": True,
                "status": {
                    "state": task_info.state,
                    "timestamp": _iso_now(),
                    "message": _make_message("agent", task_info.full_text, task_id, task_info.context_id),
                },
            })))
            await writer.drain()
            writer.close()
            return

        # Subscribe to future events
        queue: asyncio.Queue = asyncio.Queue()
        task_info.sse_queues.append(queue)

        try:
            writer.write(_sse_start_headers(origin, allowed))
            await writer.drain()

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=self.config.response_timeout)
                except asyncio.TimeoutError:
                    break

                is_final = event.get("final", False)
                state = event.get("state", "working")
                msg_obj = event.get("message")
                evt_meta = event.get("metadata")

                status_update: dict[str, Any] = {
                    "kind": "status-update",
                    "taskId": task_id,
                    "contextId": task_info.context_id,
                    "final": is_final,
                    "status": {"state": state, "timestamp": _iso_now()},
                }
                if msg_obj is not None:
                    status_update["status"]["message"] = msg_obj
                if evt_meta is not None:
                    status_update["metadata"] = evt_meta

                writer.write(_sse_data(_jsonrpc_success(req_id, status_update)))
                await writer.drain()

                if is_final:
                    break
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            if queue in task_info.sse_queues:
                task_info.sse_queues.remove(queue)
            try:
                writer.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # POST /resume, /edit_message — stubs
    # ------------------------------------------------------------------

    async def _route_stub(
        self,
        req: _HttpRequest,
        writer: asyncio.StreamWriter,
        origin: str,
        allowed: list[str],
        message: str,
    ) -> None:
        writer.write(_json_response_cors(
            {"error": message, "detail": "This endpoint is not yet implemented for nanobot"},
            501, origin, allowed,
        ))
        await writer.drain()
        writer.close()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(message: dict) -> str:
        """Extract text content from an A2A message."""
        parts = message.get("parts", [])
        texts = []
        for part in parts:
            if isinstance(part, dict):
                kind = part.get("kind", "")
                if kind == "text" and part.get("text"):
                    texts.append(part["text"])
                elif "text" in part and not kind:
                    # Fallback for simpler format
                    texts.append(part["text"])
        return "\n".join(texts).strip()
