"""PPT storyboard generation tools.

Two-phase workflow ported from workbench-agent:
  Phase 1 — ``ppt_storyboard_script``:  generate visual script (LLM)
  Phase 2 — ``ppt_storyboard_assets``:  generate images (Gemini) + narration (LLM)

LLM calls go through nanobot's ``LLMProvider``; image generation uses the
Gemini synthesis/t2i endpoint via ``GeminiImageService``.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.services.gemini_image import GeminiImageService


# ---------------------------------------------------------------------------
# Prompt templates (embedded — replaces Langfuse prompts)
# ---------------------------------------------------------------------------

SCRIPT_SYSTEM_PROMPT = """\
你是一个专业的PPT分镜视觉脚本生成器。根据用户提供的主题、大纲或文档内容，
生成一份结构化的PPT分镜视觉脚本。

对每一页PPT，输出以下内容：
- segment_id: 页码编号（如 S01, S02, …）
- title: 该页标题
- type: 页面类型（封面页/目录页/内容页/过渡页/总结页/致谢页）
- image_prompt: 该页的视觉描述（布局、核心元素、图表类型、配色建议、视觉风格）
- key_points: 该页展示的核心要点（精简 bullet points）

严格以如下 JSON 格式输出，不要附加任何其他说明：
{
  "title": "演示文稿标题",
  "contain_segment": "1",
  "storyboard": [
    {
      "segment_id": "S01",
      "title": "页面标题",
      "type": "cover",
      "image_prompt": "详细的视觉描述……",
      "key_points": ["要点1", "要点2"]
    }
  ]
}

规则：
- 保持所有页面视觉风格一致
- 配色方案专业协调
- 每页 3-5 个 bullet point
- 未指定页数时根据内容量自动规划（通常 3-15 页）
- 完整保留用户原始内容，不得删改原意
- 输出语言跟随用户输入语言
"""

ASSETS_SYSTEM_PROMPT = """\
你是一个PPT资产生成器。根据已确认的分镜视觉脚本，
为每一页生成：
1. AI绘图 prompt（英文，适用于 text-to-image 模型）
2. 口播稿（语言跟随用户输入语言）

严格以如下 JSON 格式输出，不要附加任何其他说明：
{
  "slides": [
    {
      "segment_id": "S01",
      "title": "页面标题",
      "image_prompt": "English description for AI image generation, 16:9 aspect ratio, …",
      "narration": "该页的口播演讲文案……"
    }
  ]
}

AI绘图 prompt 规则：
- 使用英文编写
- 包含：场景/布局描述、色彩方案、风格关键词、宽高比(16:9)
- 风格保持全套PPT一致性
- 避免包含文字内容（AI绘图不擅长生成文字）

口播稿规则：
- 语言自然流畅，适合口头演讲
- 每页口播时长控制在 30 秒 - 1 分钟
- 承上启下，有过渡衔接
- 包含开场白和结束语
"""


# ---------------------------------------------------------------------------
# Tool: Phase 1 — script generation
# ---------------------------------------------------------------------------


class PptStoryboardScriptTool(Tool):
    """Generate a PPT storyboard visual script from a topic / outline."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
    ):
        self._provider = provider
        self._model = model
        self._send_callback = send_callback
        self._channel = ""
        self._chat_id = ""
        # chat_id -> last generated script data (for Phase 2)
        self._session_scripts: dict[str, dict] = {}

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    def get_script(self, chat_id: str) -> dict | None:
        """Retrieve stored script for *chat_id* (used by Phase 2 tool)."""
        return self._session_scripts.get(chat_id)

    # -- Tool interface ----------------------------------------------------

    @property
    def name(self) -> str:
        return "ppt_storyboard_script"

    @property
    def description(self) -> str:
        return (
            "Generate a PPT storyboard visual script from a topic, outline, or document content. "
            "Returns a structured JSON script with page titles, visual descriptions, and key points "
            "for each slide. After user confirms, call ppt_storyboard_assets to generate images and narration."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "PPT主题或标题",
                },
                "outline": {
                    "type": "string",
                    "description": "详细大纲或文档内容（可选）",
                },
                "num_pages": {
                    "type": "integer",
                    "description": "目标页数（可选，默认自动规划）",
                },
                "style": {
                    "type": "string",
                    "description": "视觉风格偏好，如 tech / minimal / corporate（可选）",
                },
                "language": {
                    "type": "string",
                    "description": "输出语言（可选，默认跟随输入语言）",
                },
            },
            "required": ["topic"],
        }

    async def execute(
        self,
        topic: str,
        outline: str = "",
        num_pages: int | None = None,
        style: str = "",
        language: str = "",
        **kwargs: Any,
    ) -> str:
        # Build user prompt
        parts = [f"主题：{topic}"]
        if outline:
            parts.append(f"大纲/内容：\n{outline}")
        if num_pages:
            parts.append(f"目标页数：{num_pages}")
        if style:
            parts.append(f"视觉风格：{style}")
        if language:
            parts.append(f"输出语言：{language}")
        user_prompt = "\n\n".join(parts)

        # Call LLM
        messages = [
            {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = await self._provider.chat_with_retry(
            messages=messages,
            model=self._model,
            max_tokens=8192,
            temperature=0.3,
        )

        raw = response.content or ""
        if response.finish_reason == "error":
            return f"脚本生成失败：{raw[:300]}"

        # Parse JSON
        script_data = _parse_json(raw)

        # Store for Phase 2
        self._session_scripts[self._chat_id] = script_data
        logger.info(
            "PPT script generated: {} slides stored for chat_id={}",
            len(script_data.get("storyboard", [])),
            self._chat_id,
        )

        # Send XUI data message (card) if running through XUI channel
        if self._send_callback and self._channel:
            storyboards = script_data.get("storyboard", [])
            for idx, seg in enumerate(storyboards):
                params = {
                    "segmentId": seg.get("segment_id", f"S{idx + 1:02d}"),
                    "title": seg.get("title", ""),
                    "type": "wba-segment-ppt-script-generated",
                    "data": seg,
                    "showType": "11",
                    "completed": "1",
                    "ext_data": {
                        "download": True,
                        "cardLevel": 2,
                        "quote": True,
                        "autoOpen": True,
                        "canOpen": True,
                    },
                }
                biz_data = {"bizType": "common-artifact", "params": params}
                await self._send_callback(OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content="",
                    metadata={"_progress": True, "_data_message": True, "_biz_data": biz_data},
                ))

        return raw


# ---------------------------------------------------------------------------
# Tool: Phase 2 — assets generation (images + narration)
# ---------------------------------------------------------------------------


class PptStoryboardAssetsTool(Tool):
    """Generate images and narration for a confirmed PPT storyboard script."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        gemini_service: GeminiImageService | None,
        script_tool: PptStoryboardScriptTool,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
    ):
        self._provider = provider
        self._model = model
        self._gemini = gemini_service
        self._script_tool = script_tool
        self._send_callback = send_callback
        self._channel = ""
        self._chat_id = ""

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    # -- Tool interface ----------------------------------------------------

    @property
    def name(self) -> str:
        return "ppt_storyboard_assets"

    @property
    def description(self) -> str:
        return (
            "Generate AI images and narration scripts for each slide of a confirmed PPT storyboard. "
            "Must be called AFTER ppt_storyboard_script and user confirmation. "
            "Generates image prompts, sends them to Gemini for image creation, "
            "and produces narration text for each slide."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "modifications": {
                    "type": "string",
                    "description": "用户在确认时提出的修改意见（可选）",
                },
            },
            "required": [],
        }

    async def execute(self, modifications: str = "", **kwargs: Any) -> str:
        # 1. Retrieve script from Phase 1
        script_data = self._script_tool.get_script(self._chat_id)
        if not script_data:
            return "错误：未找到已生成的分镜脚本。请先调用 ppt_storyboard_script 生成脚本。"

        storyboards = script_data.get("storyboard", [])
        if not storyboards:
            return "错误：分镜脚本中没有 storyboard 条目。"

        # 2. Build prompt for assets (image prompts + narration)
        script_text = json.dumps(script_data, ensure_ascii=False, indent=2)
        user_parts = [f"以下是已确认的分镜脚本：\n```json\n{script_text}\n```"]
        if modifications:
            user_parts.append(f"用户修改意见：{modifications}")
        user_parts.append("请为每一页生成 AI 绘图 prompt 和口播稿。")

        messages = [
            {"role": "system", "content": ASSETS_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        response = await self._provider.chat_with_retry(
            messages=messages,
            model=self._model,
            max_tokens=8192,
            temperature=0.3,
        )

        raw = response.content or ""
        if response.finish_reason == "error":
            return f"资产生成失败：{raw[:300]}"

        assets_data = _parse_json(raw)
        slides = assets_data.get("slides", [])

        # 3. Generate images via Gemini for each slide
        results: list[dict] = []
        for idx, slide in enumerate(slides):
            segment_id = slide.get("segment_id", f"S{idx + 1:02d}")
            title = slide.get("title", "")
            image_prompt = slide.get("image_prompt", "")
            narration = slide.get("narration", "")

            image_url = ""
            if self._gemini and image_prompt:
                try:
                    image_url = await self._gemini.generate_image(
                        prompt=image_prompt,
                        aspect_ratio="16:9",
                    )
                    logger.info("PPT image generated for {}: {}", segment_id, (image_url or "")[:80])
                except Exception as e:
                    logger.warning("PPT image generation failed for {}: {}", segment_id, e)

            segment_result = {
                "segmentId": segment_id,
                "title": title,
                "words": narration,
                "keyframeUrl": [{"imageUrl": image_url}] if image_url else [],
                "imagePrompt": image_prompt,
                "selectedKeyframeIndex": 0,
            }
            results.append(segment_result)

            # Send XUI data message per segment
            if self._send_callback and self._channel:
                params = {
                    "type": "wba-segment-ppt-generated",
                    "segmentId": segment_id,
                    "title": title,
                    "data": segment_result,
                    "showType": "11",
                    "completed": "1",
                    "ext_data": {
                        "download": True,
                        "cardLevel": 1,
                        "quote": True,
                        "autoOpen": True,
                        "canOpen": True,
                    },
                }
                biz_data = {"bizType": "common-artifact", "params": params}
                await self._send_callback(OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content="",
                    metadata={"_progress": True, "_data_message": True, "_biz_data": biz_data},
                ))

        # 4. Summary
        ok_count = sum(1 for r in results if r.get("keyframeUrl"))
        total = len(results)
        summary_lines = [f"PPT分镜资产生成完成：共 {total} 页，{ok_count} 页成功生成图片。\n"]
        for r in results:
            img_status = "✓ 图片已生成" if r.get("keyframeUrl") else "✗ 图片生成失败"
            summary_lines.append(
                f"**{r['segmentId']} - {r['title']}**\n"
                f"  {img_status}\n"
                f"  口播稿：{(r.get('words') or '')[:80]}…\n"
            )
        return "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        first_nl = text.index("\n") if "\n" in text else 3
        text = text[first_nl + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"raw": text}
