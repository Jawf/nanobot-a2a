"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import json
import uuid
from typing import Any

import json_repair
from loguru import logger
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

# Max chars kept per tool result when collapsing into assistant content.
_TOOL_RESULT_INLINE_MAX = 4000


class CustomProvider(LLMProvider):

    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "default",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        default_headers = {
            "x-session-affinity": uuid.uuid4().hex,
            **(extra_headers or {}),
        }
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers=default_headers,
        )

    # ------------------------------------------------------------------
    # Message pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _collapse_tool_rounds(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse (assistant+tool_calls, tool_results…) into plain assistant messages.

        Many OpenAI-compatible gateways (e.g. ymcas-ai wrapping Gemini) do **not**
        pass ``thought_signature`` through.  Gemini thinking models then reject
        subsequent requests because the echoed assistant tool-call message lacks
        the signature.

        The workaround: before sending to the API, replace every completed
        tool-call round with a single assistant message whose *content* embeds
        the tool names, arguments and results in human-readable form.  The model
        receives exactly the same information; it just isn't in the formal
        ``tool_calls`` / ``tool`` message format.

        The current (last) round is collapsed too — the model already returned
        the tool call in a previous iteration of the agent loop, so it already
        "knows" it called the tool; we just need to feed the result back.
        """
        out: list[dict[str, Any]] = []
        i = 0
        n = len(messages)
        while i < n:
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                tc_ids = {
                    tc["id"] for tc in tool_calls
                    if isinstance(tc, dict) and tc.get("id")
                }

                # Gather consecutive tool-result messages that belong to this round.
                j = i + 1
                results: dict[str, str] = {}
                while j < n and messages[j].get("role") == "tool":
                    tid = messages[j].get("tool_call_id")
                    if tid in tc_ids:
                        results[tid] = messages[j].get("content", "")
                    j += 1

                # Build a plain-text summary.
                parts: list[str] = []
                if msg.get("content"):
                    parts.append(msg["content"])
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "{}")
                    res = results.get(tc.get("id", ""), "(no result)")
                    if isinstance(res, str) and len(res) > _TOOL_RESULT_INLINE_MAX:
                        res = res[:_TOOL_RESULT_INLINE_MAX] + "…(truncated)"
                    parts.append(f"[Called tool `{name}` with args: {args}]\n[Result: {res}]")

                out.append({
                    "role": "assistant",
                    "content": "\n\n".join(parts) or "(tool executed)",
                })
                i = j  # skip past consumed tool-result messages
                continue

            # Skip orphan tool-result messages whose assistant round was
            # already collapsed (or never present).
            if msg.get("role") == "tool":
                i += 1
                continue

            out.append(msg)
            i += 1

        return out

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None,
                   tool_choice: str | dict[str, Any] | None = None) -> LLMResponse:
        clean = self._sanitize_empty_content(messages)

        # Collapse prior tool-call rounds so Gemini thinking models don't
        # reject the request for missing thought_signature.
        clean = self._collapse_tool_rounds(clean)

        # Defensive: strip non-standard fields and tool artefacts.
        # Many Gemini gateways (ymcas-ai) choke on:
        #   - tool_calls / tool role  (missing thought_signature)
        #   - reasoning_content / thinking_blocks  (thinking model metadata)
        # Keep only the standard OpenAI Chat Completion fields per role.
        _ALLOWED_KEYS = {
            "system":    {"role", "content", "name"},
            "user":      {"role", "content", "name"},
            "assistant": {"role", "content"},  # tool_calls already collapsed
            "tool":      {"role", "content", "tool_call_id", "name"},
        }
        sanitized: list[dict[str, Any]] = []
        for msg in clean:
            role = msg.get("role", "")
            if role == "tool":
                continue  # orphan tool-result
            if role == "assistant" and msg.get("tool_calls"):
                logger.warning(
                    "tool_calls survived collapse! Stripping from message: {}",
                    (msg.get("content") or "")[:100],
                )
            allowed = _ALLOWED_KEYS.get(role, {"role", "content"})
            sanitized.append({k: v for k, v in msg.items() if k in allowed})
        clean = sanitized

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": clean,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto")
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            err_msg = str(e)
            logger.error("Custom provider API error (model={}):\n{}", kwargs.get("model"), err_msg)
            return LLMResponse(content=f"Error: {err_msg}", finish_reason="error")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(self, response: Any) -> LLMResponse:
        if not response.choices:
            return LLMResponse(
                content="Error: API returned empty choices. This may indicate a temporary service issue or an invalid model response.",
                finish_reason="error"
            )
        choice = response.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)
            tool_calls.append(ToolCallRequest(
                id=tc.id, name=tc.function.name, arguments=args,
            ))
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model
