"""Claude CLI integration tool.

Invokes the `claude` CLI (Claude Code) in non-interactive print mode and
relays the streamed response back to the originating chat channel.

Multi-turn conversation is supported: each chat context (channel + chat_id)
keeps its own Claude CLI session ID so follow-up messages continue the
same conversation.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage

# Characters accumulated before sending a progress chunk to the chat channel.
# Keep this large enough to avoid spamming, small enough to feel responsive.
_PROGRESS_CHUNK = 600


class ClaudeCliTool(Tool):
    """Invoke Claude CLI and stream the response to the current chat channel.

    The tool uses ``claude --print --output-format stream-json`` so output
    arrives as newline-delimited JSON events.  Text is extracted from
    ``assistant`` events and streamed to the channel as progress messages.
    The ``result`` event at the end provides the canonical full response and
    the session ID that is stored for the next turn.
    """

    _MAX_TIMEOUT = 600

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        claude_bin: str = "claude",
        timeout: int = 300,
    ) -> None:
        self._send_callback = send_callback
        self._claude_bin = claude_bin
        self._timeout = timeout
        self._channel: str = ""
        self._chat_id: str = ""
        # session_id per "channel:chat_id" for multi-turn conversations
        self._sessions: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Context injection (called by AgentLoop._set_tool_context)
    # ------------------------------------------------------------------

    def set_context(self, channel: str, chat_id: str) -> None:
        """Record current routing context so progress messages reach the right chat."""
        self._channel = channel
        self._chat_id = chat_id

    # ------------------------------------------------------------------
    # Tool metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "claude_cli"

    @property
    def description(self) -> str:
        return (
            "Invoke the Claude CLI (claude-code) with a prompt and stream "
            "the response directly to the chat channel. "
            "Supports multi-turn conversation — follow-up calls automatically "
            "continue the same Claude session for this chat."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The message or task to send to Claude CLI.",
                },
                "new_session": {
                    "type": "boolean",
                    "description": (
                        "Start a brand-new Claude conversation, discarding any "
                        "previous session for this chat. Default: false."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait for a response (10-600, default 300).",
                    "minimum": 10,
                    "maximum": 600,
                },
            },
            "required": ["prompt"],
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        prompt: str,
        new_session: bool = False,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        # Resolve binary path
        resolved = shutil.which(self._claude_bin)
        if not resolved:
            return (
                "Error: claude CLI not found on PATH. "
                "Install it with: npm install -g @anthropic-ai/claude-code"
            )

        effective_timeout = min(timeout or self._timeout, self._MAX_TIMEOUT)
        chat_key = f"{self._channel}:{self._chat_id}"

        # Handle session continuity
        if new_session:
            self._sessions.pop(chat_key, None)
        prev_session = self._sessions.get(chat_key)

        # Build subprocess command.
        # IMPORTANT: --print / -p takes the prompt as its value, so it MUST
        # come immediately before the prompt string.  Placing --output-format
        # after --print would cause --print to consume "--output-format" as
        # its value, which results in a "conflicting options" error.
        # --output-format=stream-json requires --verbose when combined with --print
        cmd: list[str] = [resolved, "--output-format", "stream-json", "--verbose"]
        if prev_session:
            cmd += ["--resume", prev_session]
        cmd += ["--print", prompt]

        logger.info(
            "claude_cli: chat_key={}, session={}, timeout={}s",
            chat_key,
            prev_session or "new",
            effective_timeout,
        )

        # Notify the user that we are working on it
        await self._send_progress(
            "🤖 **Claude CLI** 正在处理你的请求，请稍候…"
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return (
                "Error: claude binary not found. "
                "Install Claude Code: npm install -g @anthropic-ai/claude-code"
            )
        except Exception as exc:
            return f"Error: failed to start claude CLI: {exc}"

        try:
            final_text, new_session_id, cli_error = await asyncio.wait_for(
                self._read_stream(process),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return f"Error: Claude CLI timed out after {effective_timeout}s"

        # Collect stderr for error reporting
        try:
            stderr_bytes = await asyncio.wait_for(
                process.stderr.read(), timeout=3.0  # type: ignore[union-attr]
            )
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            stderr_text = ""

        await process.wait()

        # Persist session ID for next turn
        if new_session_id:
            self._sessions[chat_key] = new_session_id
            logger.info("claude_cli: saved session_id={} for {}", new_session_id, chat_key)

        if cli_error:
            err_msg = f"❌ Claude CLI 返回错误：{cli_error}"
            await self._send_progress(err_msg)
            return f"Claude CLI error: {cli_error}"

        if not final_text:
            if stderr_text:
                return f"Claude CLI error (stderr): {stderr_text[:800]}"
            return "Claude CLI returned no output"

        # Send the complete response as a normal (non-progress) message so it
        # reaches the user even when send_progress is disabled in config.
        await self._send_message(final_text)

        session_hint = f" (session: {new_session_id[:8]}…)" if new_session_id else ""
        return (
            f"Claude CLI response delivered to chat{session_hint}.\n\n"
            f"Response preview (first 400 chars):\n{final_text[:400]}"
        )

    # ------------------------------------------------------------------
    # Stream parsing
    # ------------------------------------------------------------------

    async def _read_stream(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[str, str | None, str | None]:
        """Read stdout JSONL events.

        Returns (full_text, session_id, error_message).
        """
        assert process.stdout is not None

        text_parts: list[str] = []
        pending: list[str] = []
        pending_len: int = 0
        session_id: str | None = None
        error_msg: str | None = None

        async def flush_pending() -> None:
            nonlocal pending_len
            if pending and self._send_callback and self._channel:
                chunk = "".join(pending)
                await self._send_callback(
                    OutboundMessage(
                        channel=self._channel,
                        chat_id=self._chat_id,
                        content=chunk,
                        metadata={"_progress": True},
                    )
                )
            pending.clear()
            pending_len = 0

        async for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("claude_cli: skipping non-JSON line: {}", line[:120])
                continue

            etype = event.get("type", "")

            if etype == "assistant":
                # Extract text blocks from the assistant message
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        txt = block.get("text", "")
                        if txt:
                            text_parts.append(txt)
                            pending.append(txt)
                            pending_len += len(txt)
                            if pending_len >= _PROGRESS_CHUNK:
                                await flush_pending()

            elif etype == "result":
                session_id = event.get("session_id")
                if event.get("is_error"):
                    error_msg = str(event.get("result", "unknown error"))
                else:
                    # Use the canonical result text if it differs from what we accumulated
                    canonical = event.get("result", "")
                    if canonical and not text_parts:
                        text_parts.append(canonical)
                # Flush any remaining pending text
                await flush_pending()

        return "".join(text_parts).strip(), session_id, error_msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_progress(self, content: str) -> None:
        """Send a progress/status message (filtered by send_progress config)."""
        if self._send_callback and self._channel and self._chat_id:
            await self._send_callback(
                OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content=content,
                    metadata={"_progress": True},
                )
            )

    async def _send_message(self, content: str) -> None:
        """Send a regular (non-progress) message to the chat channel."""
        if self._send_callback and self._channel and self._chat_id:
            await self._send_callback(
                OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content=content,
                )
            )
