"""Comic storyboard generation tools.

Four-phase workflow for converting novels into dynamic comic videos:
  Phase 1 — ``comic_storyboard_script``:  parse novel → structured storyboard JSON (LLM)
  Phase 2 — ``comic_storyboard_images``:  generate panel images (Gemini T2I)
  Phase 3 — ``comic_storyboard_tts``:     generate narration/dialogue audio (TTS)
  Phase 4 — ``comic_storyboard_video``:   composite images + audio → MP4 (ffmpeg)

LLM calls go through nanobot's ``LLMProvider``; image generation uses the
Gemini synthesis/t2i endpoint via ``GeminiImageService``; TTS uses ``TTSService``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import json_repair

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.services.gemini_image import GeminiImageService
    from nanobot.services.tts import TTSService


# ---------------------------------------------------------------------------
# Session persistence helpers — survive process restarts between phases
# ---------------------------------------------------------------------------

_SESSIONS_SUBDIR = ".sessions"


def _safe_chat_id(chat_id: str) -> str:
    """Sanitize chat_id for use as a filesystem directory name."""
    return chat_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def _save_session_data(base: Path, chat_id: str, name: str, data: dict) -> None:
    """Persist session data to disk under ``base/.sessions/{chat_id}/{name}.json``."""
    d = base / _SESSIONS_SUBDIR / _safe_chat_id(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_session_data(base: Path, chat_id: str, name: str) -> dict | None:
    """Load session data from disk; returns *None* if missing or corrupt."""
    f = base / _SESSIONS_SUBDIR / _safe_chat_id(chat_id) / f"{name}.json"
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _clear_session_data(base: Path, chat_id: str) -> None:
    """Remove all persisted session files for *chat_id*."""
    d = base / _SESSIONS_SUBDIR / _safe_chat_id(chat_id)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

COMIC_SCRIPT_SYSTEM_PROMPT = """\
你是一个专业的动态漫画分镜脚本生成器。根据用户提供的小说内容（可来自爆款网文创作技能的输出），
生成一份结构化的动态漫画分镜脚本。

输入可能包含：
- 小说章节大纲（章节号+标题+核心事件+爽点类型）
- 角色描述（人设标签、外貌特征）
- 章节正文（2000-4000字/章）
- 套路画像、卖点设计等元信息

你的任务是将小说内容转化为可直接用于漫画制作的分镜脚本，严格遵守以下规则：

## 爆款动态漫规则
- **前三秒/前3镜定生死**：第一集前3镜必须出现强视觉或强冲突
- **单集结构**：Hook → 铺垫/冲突升级 → 本集高潮 → 悬念收尾
- **节奏**：快切、少长对白；长对白拆成多面板+表情/动作
- **每集至少一个"可截图传播"的高潮画面**
- **结尾必须有悬念或下集期待**

## 专业分镜能力
- 景别交替：远景→中景→近景/特写
- 构图与视线引导：三分法、留白
- 分格节奏：紧张段多小格快切，高潮大格/跨格
- 对白精简：小说长句改短句金句，每格1-2句

## 输出格式
严格以如下 JSON 格式输出，不要附加任何其他说明：
{
  "title": "漫画标题",
  "total_episodes": 5,
  "aspect_ratio": "9:16",
  "style": "国漫写实",
  "characters": [
    {
      "name": "角色名",
      "description": "角色简介与人设标签",
      "gender": "male/female",
      "visual_tags": ["黑发", "红色长袍", "剑客", "冷峻"]
    }
  ],
  "episodes": [
    {
      "episode_id": "E01",
      "title": "集标题",
      "novel_reference": "对应小说第X章",
      "hook": "前3镜钩子描述",
      "suspense_ending": "悬念收尾文字",
      "emotion_tone": "燃/虐/爽/悬疑",
      "panels": [
        {
          "panel_id": "E01P01",
          "shot_type": "特写/中景/远景/俯视/仰视",
          "image_prompt": "English prompt for AI image generation. Include character visual tags, scene description, composition, lighting, style keywords. Must be detailed enough for T2I model.",
          "characters": ["角色A"],
          "dialogue": "角色A：台词内容",
          "narration": "旁白/画外音文字（用于TTS朗读）",
          "motion_suggestion": "推镜至面部, 0.5s",
          "duration_s": 3.0,
          "is_climax": false
        }
      ]
    }
  ]
}

## image_prompt 规则
- 必须用英文编写
- 包含：角色外貌标签（从characters中提取）、场景描述、构图指示、光影氛围、画风关键词
- 保持全集角色外貌描述一致（每次都包含角色的visual_tags）
- **同一集内的场景环境描述必须保持一致**（如"废弃仓库"不能突然变成"古风宫殿"）
- **严禁在 image_prompt 中包含任何文字/数字/字母**，画面中不能出现文字
- **必须明确画风关键词**（如 manga style, illustration, comic art），避免生成写实照片风格
- 包含宽高比指示（如 vertical 9:16 composition）

## dialogue 格式规则
- 必须以 "角色名：台词" 或 "角色名(情绪)：台词" 格式书写，如 "林夜(冰冷)：拿了我的东西"
- 旁白用 "旁白：" 或 "旁白(语气)：" 开头
- 音效用 "音效：" 开头（如 "音效：轰隆隆的雷声"）
- 系统提示用 "系统音：" 开头
- 每个面板的 dialogue 应只有一位主要说话人（多人对话拆分到不同面板）
- 这个格式用于自动分配不同角色的 TTS 音色，必须严格遵守

## duration_s 规则
- 对白格：根据对白字数估算朗读时长（中文约每秒3-4字）
- 旁白格：根据旁白字数估算
- 纯画面格（无对白无旁白）：2-4秒
- 高潮定格：额外+1-2秒
"""

COMIC_IMAGES_REFINE_PROMPT = """\
你是一个漫画图片prompt优化器。根据已有的分镜脚本，为每个面板优化AI绘图prompt。

要求：
- 保持角色外貌在全集中一致
- 添加适合的光影、氛围、构图细节
- 确保prompt适用于text-to-image模型
- 用英文输出

严格以如下 JSON 格式输出：
{
  "panels": [
    {
      "panel_id": "E01P01",
      "image_prompt": "Optimized English prompt..."
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Phase 1: Script generation
# ---------------------------------------------------------------------------


class ComicStoryboardScriptTool(Tool):
    """Generate a comic storyboard script from novel input."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        sessions_base: Path | None = None,
    ):
        self._provider = provider
        self._model = model
        self._send_callback = send_callback
        self._sessions_base = sessions_base
        self._channel = ""
        self._chat_id = ""
        self._session_scripts: dict[str, dict] = {}

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    def get_script(self, chat_id: str) -> dict | None:
        """Retrieve stored script for *chat_id* (used by subsequent phase tools).

        Checks in-memory cache first, then falls back to disk.
        """
        data = self._session_scripts.get(chat_id)
        if data is None and self._sessions_base:
            data = _load_session_data(self._sessions_base, chat_id, "script")
            if data is not None:
                self._session_scripts[chat_id] = data
                logger.info("Comic script restored from disk for chat_id={}", chat_id)
        return data

    @property
    def name(self) -> str:
        return "comic_storyboard_script"

    @property
    def description(self) -> str:
        return (
            "Generate a dynamic comic storyboard script from novel text. "
            "Accepts novel content (chapters, outlines, character descriptions — "
            "e.g. output from the bestseller-novel skill) and produces a structured JSON "
            "storyboard with episodes, panels, image prompts, dialogue, and narration. "
            "After user confirms, call comic_storyboard_images to generate panel images."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "novel_text": {
                    "type": "string",
                    "description": "小说内容（章节大纲+角色描述+章节正文，可来自 bestseller-novel 技能输出）",
                },
                "platform": {
                    "type": "string",
                    "description": "目标平台（抖音/快手/B站/快看/微短剧），默认抖音",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "目标集数（可选，默认根据内容量自动规划）",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "画面比例（9:16竖屏/16:9横屏），默认9:16",
                },
                "style": {
                    "type": "string",
                    "description": "画风（国漫写实/日漫/Q版/水墨），默认国漫写实",
                },
            },
            "required": ["novel_text"],
        }

    async def execute(
        self,
        novel_text: str,
        platform: str = "抖音",
        num_episodes: int | None = None,
        aspect_ratio: str = "9:16",
        style: str = "国漫写实",
        **kwargs: Any,
    ) -> str:
        # Notify user immediately — script generation can take 1-3 minutes
        if self._send_callback and self._channel:
            ep_hint = f"，目标 {num_episodes} 集" if num_episodes else ""
            await self._send_callback(OutboundMessage(
                channel=self._channel,
                chat_id=self._chat_id,
                content=f"🎬 正在生成动漫分镜脚本{ep_hint}，预计需要 1-3 分钟，请稍候…",
                metadata={"_progress": True},
            ))

        # Cap input length to prevent excessive token usage / provider timeouts.
        # ~6000 chars ≈ 3000 Chinese chars ≈ 2-3 chapters, enough for 3-5 episodes.
        _MAX_NOVEL_CHARS = 6000
        if len(novel_text) > _MAX_NOVEL_CHARS:
            logger.info(
                "Novel text truncated from {} to {} chars for comic script generation",
                len(novel_text), _MAX_NOVEL_CHARS,
            )
            novel_text = novel_text[:_MAX_NOVEL_CHARS] + "\n\n[...（内容已截断，请根据以上内容生成脚本）]"

        parts = [f"目标平台：{platform}", f"画面比例：{aspect_ratio}", f"画风：{style}"]
        if num_episodes:
            parts.append(f"目标集数：{num_episodes}")
        parts.append(f"小说内容：\n{novel_text}")
        user_prompt = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": COMIC_SCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = await self._provider.chat_with_retry(
            messages=messages,
            model=self._model,
            max_tokens=16384,
            temperature=0.3,
        )

        raw = response.content or ""
        if response.finish_reason == "error":
            return f"分镜脚本生成失败：{raw[:300]}"

        script_data = _parse_json(raw)

        # Store for subsequent phases (memory + disk)
        self._session_scripts[self._chat_id] = script_data
        if self._sessions_base:
            _save_session_data(self._sessions_base, self._chat_id, "script", script_data)
            logger.info(
                "Comic script session saved to disk: {}",
                self._sessions_base / _SESSIONS_SUBDIR / _safe_chat_id(self._chat_id) / "script.json",
            )

        episodes = script_data.get("episodes", [])
        total_panels = sum(len(ep.get("panels", [])) for ep in episodes)
        characters = [c.get("name", "?") for c in script_data.get("characters", [])]
        logger.info(
            "Comic script generated: {} episodes, {} panels, characters={} for chat_id={}",
            len(episodes), total_panels, characters, self._chat_id,
        )

        # Send XUI data messages per episode
        if self._send_callback and self._channel:
            for idx, ep in enumerate(episodes):
                params = {
                    "segmentId": ep.get("episode_id", f"E{idx + 1:02d}"),
                    "title": ep.get("title", ""),
                    "type": "wba-segment-comic-script-generated",
                    "data": ep,
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
# Phase 2: Image generation
# ---------------------------------------------------------------------------


class ComicStoryboardImagesTool(Tool):
    """Generate AI images for comic panels using Gemini T2I."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        gemini_service: GeminiImageService | None,
        script_tool: ComicStoryboardScriptTool,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        output_dir: Path | None = None,
    ):
        self._provider = provider
        self._model = model
        self._gemini = gemini_service
        self._script_tool = script_tool
        self._send_callback = send_callback
        self._channel = ""
        self._chat_id = ""
        self._session_images: dict[str, dict[str, str]] = {}  # chat_id -> {panel_id: local_path}
        if output_dir is None:
            fallback = Path(tempfile.gettempdir()) / "nanobot_comic_images"
            logger.warning(
                "ComicStoryboardImagesTool: output_dir not set; images will go to system temp: {}. "
                "Pass output_dir=workspace/comic_assets/images when registering.",
                fallback,
            )
            self._output_dir = fallback
            self._sessions_base: Path | None = None
        else:
            self._output_dir = output_dir
            self._sessions_base = output_dir.parent  # workspace/comic_assets
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    def get_images(self, chat_id: str) -> dict[str, str] | None:
        """Retrieve stored images for *chat_id*.

        Checks in-memory cache first, then falls back to disk.
        """
        data = self._session_images.get(chat_id)
        if data is None and self._sessions_base:
            data = _load_session_data(self._sessions_base, chat_id, "images")
            if data is not None:
                self._session_images[chat_id] = data
                logger.info("Comic images restored from disk for chat_id={}", chat_id)
        return data

    @property
    def name(self) -> str:
        return "comic_storyboard_images"

    @property
    def description(self) -> str:
        return (
            "Generate AI images for each panel of a confirmed comic storyboard script. "
            "Must be called AFTER comic_storyboard_script and user confirmation. "
            "Uses Gemini T2I to generate panel images based on image_prompt in the script."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "modifications": {
                    "type": "string",
                    "description": "用户修改意见（可选）",
                },
                "episode_filter": {
                    "type": "string",
                    "description": "仅生成指定集的图片（如 E01），不指定则全部生成",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        modifications: str = "",
        episode_filter: str = "",
        **kwargs: Any,
    ) -> str:
        logger.info(
            "[Images] Phase 2 started: chat_id={}, output_dir={}, episode_filter={}",
            self._chat_id, self._output_dir, episode_filter or "(all)",
        )
        script_data = self._script_tool.get_script(self._chat_id)
        if not script_data:
            logger.warning("[Images] No script found for chat_id={}", self._chat_id)
            return "错误：未找到已生成的分镜脚本。请先调用 comic_storyboard_script 生成脚本。"

        if not self._gemini:
            logger.warning("[Images] Gemini service not configured")
            return "错误：Gemini 图片生成服务未配置。请在 config.json 中配置 tools.gemini_image。"

        episodes = script_data.get("episodes", [])
        aspect_ratio = script_data.get("aspect_ratio", "9:16")
        images: dict[str, str] = self._session_images.get(self._chat_id, {})

        total_panels = sum(len(ep.get("panels", [])) for ep in episodes)
        ep_label = f"第 {episode_filter} 集" if episode_filter else f"全部 {len(episodes)} 集"
        if self._send_callback and self._channel:
            await self._send_callback(OutboundMessage(
                channel=self._channel,
                chat_id=self._chat_id,
                content=f"🖼 开始生成漫画面板图片（{ep_label}，共 {total_panels} 张），每张约 5-15 秒…",
                metadata={"_progress": True},
            ))

        # Build a global prompt suffix for style/character consistency
        prompt_suffix = _build_prompt_suffix(script_data)

        ok_count = 0
        fail_count = 0

        for ep in episodes:
            ep_id = ep.get("episode_id", "")
            if episode_filter and ep_id != episode_filter:
                continue

            # Extract a per-episode scene anchor from the first panel's prompt
            ep_scene_anchor = _extract_scene_anchor(ep)

            for panel in ep.get("panels", []):
                panel_id = panel.get("panel_id", "")
                image_prompt = panel.get("image_prompt", "")
                if not image_prompt:
                    continue

                # Enhance prompt: scene anchor + character tags + anti-text
                image_prompt = _enhance_image_prompt(
                    image_prompt, ep_scene_anchor, prompt_suffix,
                )

                # Retry once on failure (transient API errors are common with T2I)
                image_url = None
                last_err = None
                for attempt in range(2):
                    try:
                        image_url = await self._gemini.generate_image(
                            prompt=image_prompt,
                            aspect_ratio=aspect_ratio,
                        )
                        break
                    except Exception as e:
                        last_err = e
                        if attempt == 0:
                            logger.info("Comic image retry for {} after: {}", panel_id, e)
                            await asyncio.sleep(1)

                if image_url:
                    # Save image to disk immediately to avoid URL expiration during Phase 4.
                    ep_dir = self._output_dir / ep_id
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    img_dest = ep_dir / panel_id
                    try:
                        local_path = await _download_image(image_url, img_dest)
                        images[panel_id] = str(local_path)
                        logger.info("Comic image saved for {}: {}", panel_id, local_path)
                    except Exception as save_err:
                        logger.warning("Comic image save failed for {}, storing URL: {}", panel_id, save_err)
                        images[panel_id] = image_url
                    ok_count += 1
                else:
                    fail_count += 1
                    logger.warning("Comic image generation failed for {}: {}", panel_id, last_err)

                # Send XUI progress per panel
                if self._send_callback and self._channel:
                    panel_result = {
                        "panelId": panel_id,
                        "episodeId": ep_id,
                        "imageUrl": image_url or "",  # XUI uses original URL for display
                        "imagePrompt": image_prompt,
                    }
                    params = {
                        "segmentId": panel_id,
                        "title": f"{ep_id} - {panel.get('shot_type', '')}",
                        "type": "wba-segment-comic-panel-generated",
                        "data": panel_result,
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

        self._session_images[self._chat_id] = images
        if self._sessions_base:
            _save_session_data(self._sessions_base, self._chat_id, "images", images)

        # Notify user when ALL images failed
        if ok_count == 0 and fail_count > 0 and self._send_callback and self._channel:
            await self._send_callback(OutboundMessage(
                channel=self._channel,
                chat_id=self._chat_id,
                content=(
                    f"⚠️ 漫画图片生成全部失败（共 {fail_count} 张）。"
                    "可能原因：图片生成服务暂时不可用，请稍后重试或检查 Gemini 配置。"
                ),
            ))

        summary = f"漫画面板图片生成完成：成功 {ok_count} 张，失败 {fail_count} 张。\n"
        for ep in episodes:
            ep_id = ep.get("episode_id", "")
            if episode_filter and ep_id != episode_filter:
                continue
            summary += f"\n**{ep_id} - {ep.get('title', '')}**\n"
            for panel in ep.get("panels", []):
                pid = panel.get("panel_id", "")
                status = "✓" if pid in images and images[pid] else "✗"
                summary += f"  {status} {pid} ({panel.get('shot_type', '')})\n"

        summary += f"\n图片保存目录：{self._output_dir}"
        return summary


# ---------------------------------------------------------------------------
# Phase 3: TTS generation
# ---------------------------------------------------------------------------


class ComicStoryboardTTSTool(Tool):
    """Generate TTS audio for comic panel narration and dialogue."""

    def __init__(
        self,
        tts_service: TTSService | None,
        script_tool: ComicStoryboardScriptTool,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        sessions_base: Path | None = None,
    ):
        self._tts = tts_service
        self._script_tool = script_tool
        self._send_callback = send_callback
        self._sessions_base = sessions_base
        self._channel = ""
        self._chat_id = ""
        self._session_audio: dict[str, dict[str, str]] = {}  # chat_id -> {panel_id: audio_path}

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    def get_audio(self, chat_id: str) -> dict[str, str] | None:
        """Retrieve stored audio paths for *chat_id*.

        Checks in-memory cache first, then falls back to disk.
        """
        data = self._session_audio.get(chat_id)
        if data is None and self._sessions_base:
            data = _load_session_data(self._sessions_base, chat_id, "audio")
            if data is not None:
                self._session_audio[chat_id] = data
                logger.info("Comic audio restored from disk for chat_id={}", chat_id)
        return data

    @property
    def name(self) -> str:
        return "comic_storyboard_tts"

    @property
    def description(self) -> str:
        return (
            "Generate TTS audio for each panel's narration and dialogue in the comic storyboard. "
            "Must be called AFTER comic_storyboard_script (and optionally comic_storyboard_images). "
            "Produces MP3 audio files for each panel."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "episode_filter": {
                    "type": "string",
                    "description": "仅为指定集生成语音（如 E01），不指定则全部生成",
                },
                "voice": {
                    "type": "string",
                    "description": "TTS音色覆盖（如 zh-CN-YunxiNeural），不指定则使用默认配置",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        episode_filter: str = "",
        voice: str = "",
        **kwargs: Any,
    ) -> str:
        logger.info(
            "[TTS] Phase 3 started: chat_id={}, output_dir={}, episode_filter={}",
            self._chat_id, getattr(self._tts, "output_dir", "N/A"), episode_filter or "(all)",
        )
        script_data = self._script_tool.get_script(self._chat_id)
        if not script_data:
            logger.warning("[TTS] No script found for chat_id={}", self._chat_id)
            return "错误：未找到已生成的分镜脚本。请先调用 comic_storyboard_script 生成脚本。"

        if not self._tts:
            logger.warning("[TTS] TTS service not configured")
            return "错误：TTS 服务未配置。请在 config.json 中配置 tools.tts。"

        episodes = script_data.get("episodes", [])
        audio_map: dict[str, str] = self._session_audio.get(self._chat_id, {})

        # Build per-speaker voice map (auto-assigned from characters or explicit)
        voice_map = _build_voice_map(script_data)
        logger.info("Comic TTS voice map: {}", voice_map)

        ok_count = 0
        fail_count = 0
        skip_count = 0

        for ep in episodes:
            ep_id = ep.get("episode_id", "")
            if episode_filter and ep_id != episode_filter:
                continue

            for panel in ep.get("panels", []):
                panel_id = panel.get("panel_id", "")

                # Combine dialogue and narration for TTS
                tts_parts = []
                narration = panel.get("narration", "")
                dialogue = panel.get("dialogue", "")

                # Detect speaker + emotion from dialogue for voice/param selection
                speaker = ""
                emotion = ""
                if dialogue:
                    speaker, emotion, clean_dialogue = _detect_speaker_emotion(dialogue)
                    if clean_dialogue:
                        tts_parts.append(clean_dialogue)
                if narration:
                    if not speaker:
                        speaker = "旁白"
                    tts_parts.append(narration)

                tts_text = " ".join(tts_parts).strip()
                if not tts_text:
                    skip_count += 1
                    continue

                # Select voice: explicit override > voice_map by speaker > default
                panel_voice = voice or voice_map.get(speaker) or None

                # Resolve emotion: explicit tag > infer from speaker+text > None
                if speaker == "旁白":
                    resolved_emotion = None
                elif emotion:
                    resolved_emotion = emotion
                else:
                    resolved_emotion = _infer_emotion(speaker, tts_text)

                logger.debug("TTS {}: speaker={}, emotion={}->{}, voice={}",
                             panel_id, speaker, emotion or "(none)", resolved_emotion, panel_voice)

                try:
                    audio_path = await self._tts.synthesize(
                        text=tts_text,
                        filename=panel_id,
                        voice=panel_voice,
                        emotion=resolved_emotion,
                    )
                    audio_map[panel_id] = audio_path
                    ok_count += 1

                    # Update panel duration to match actual audio length + breathing room
                    audio_dur = await _get_audio_duration(audio_path)
                    if audio_dur > 0:
                        panel["duration_s"] = round(max(audio_dur + 0.8, 2.5), 1)

                    logger.info("Comic TTS generated for {}: {} (speaker={}, emotion={}, dur={:.1f}s)",
                                panel_id, audio_path, speaker, resolved_emotion, panel.get("duration_s", 0))
                except Exception as e:
                    fail_count += 1
                    logger.warning("Comic TTS failed for {}: {}", panel_id, e)

            # Send XUI progress per episode
            if self._send_callback and self._channel:
                ep_panels = [p.get("panel_id", "") for p in ep.get("panels", [])]
                ep_ok = sum(1 for pid in ep_panels if pid in audio_map)
                params = {
                    "segmentId": ep_id,
                    "title": f"{ep_id} TTS: {ep_ok}/{len(ep_panels)}",
                    "type": "wba-segment-comic-tts-generated",
                    "data": {
                        "episodeId": ep_id,
                        "totalPanels": len(ep_panels),
                        "generatedPanels": ep_ok,
                    },
                    "showType": "11",
                    "completed": "1",
                    "ext_data": {"cardLevel": 1, "canOpen": True},
                }
                biz_data = {"bizType": "common-artifact", "params": params}
                await self._send_callback(OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content="",
                    metadata={"_progress": True, "_data_message": True, "_biz_data": biz_data},
                ))

        self._session_audio[self._chat_id] = audio_map
        if self._sessions_base:
            _save_session_data(self._sessions_base, self._chat_id, "audio", audio_map)

        return (
            f"TTS语音生成完成：成功 {ok_count} 个面板，失败 {fail_count} 个，"
            f"跳过 {skip_count} 个（无文字）。\n"
            f"音频文件目录：{self._tts.output_dir}"
        )


# ---------------------------------------------------------------------------
# Phase 4: Video composition
# ---------------------------------------------------------------------------


class ComicStoryboardVideoTool(Tool):
    """Composite panel images + TTS audio into per-episode video via ffmpeg."""

    def __init__(
        self,
        script_tool: ComicStoryboardScriptTool,
        images_tool: ComicStoryboardImagesTool,
        tts_tool: ComicStoryboardTTSTool,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        output_dir: Path | None = None,
    ):
        self._script_tool = script_tool
        self._images_tool = images_tool
        self._tts_tool = tts_tool
        self._send_callback = send_callback
        self._channel = ""
        self._chat_id = ""
        if output_dir is None:
            # Fallback only — callers should always pass the workspace-relative output_dir.
            fallback = Path(tempfile.gettempdir()) / "nanobot_comic_video"
            logger.warning(
                "ComicStoryboardVideoTool: output_dir not set; files will go to system temp: {}. "
                "Pass output_dir=workspace/comic_assets/video when registering the tool.",
                fallback,
            )
            self._output_dir = fallback
        else:
            self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    @property
    def name(self) -> str:
        return "comic_storyboard_video"

    @property
    def description(self) -> str:
        return (
            "Composite panel images and TTS audio into per-episode MP4 video using ffmpeg. "
            "Must be called AFTER comic_storyboard_images and comic_storyboard_tts. "
            "Produces one video file per episode."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "episode_filter": {
                    "type": "string",
                    "description": "仅合成指定集的视频（如 E01），不指定则全部合成",
                },
            },
            "required": [],
        }

    async def execute(self, episode_filter: str = "", **kwargs: Any) -> str:
        logger.info(
            "[Video] Phase 4 started: chat_id={}, output_dir={}, episode_filter={}",
            self._chat_id, self._output_dir, episode_filter or "(all)",
        )
        script_data = self._script_tool.get_script(self._chat_id)
        if not script_data:
            logger.warning("[Video] No script found for chat_id={}", self._chat_id)
            return "错误：未找到分镜脚本。请先完成前三个阶段。"

        images = self._images_tool.get_images(self._chat_id)
        if not images:
            logger.warning("[Video] No images found for chat_id={}", self._chat_id)
            return "错误：未找到面板图片。请先调用 comic_storyboard_images。"

        audio_map = self._tts_tool.get_audio(self._chat_id) or {}
        logger.info("[Video] Resources: {} images, {} audio clips", len(images), len(audio_map))

        episodes = script_data.get("episodes", [])
        aspect_ratio = script_data.get("aspect_ratio", "9:16")
        results: list[dict] = []

        for ep in episodes:
            ep_id = ep.get("episode_id", "")
            if episode_filter and ep_id != episode_filter:
                continue

            panels = ep.get("panels", [])
            if not panels:
                continue

            try:
                video_path = await _compose_episode_video(
                    episode_id=ep_id,
                    panels=panels,
                    images=images,
                    audio_map=audio_map,
                    output_dir=self._output_dir,
                    aspect_ratio=aspect_ratio,
                )
                results.append({"episode_id": ep_id, "video_path": video_path, "success": True})
                logger.info("Comic video composed for {}: {}", ep_id, video_path)
            except Exception as e:
                results.append({"episode_id": ep_id, "error": str(e), "success": False})
                logger.exception("[Video] Composition failed for {}", ep_id)

            # Send XUI progress per episode
            if self._send_callback and self._channel:
                r = results[-1]
                params = {
                    "segmentId": ep_id,
                    "title": f"{ep_id} {'✓' if r['success'] else '✗'}",
                    "type": "wba-segment-comic-video-generated",
                    "data": r,
                    "showType": "11",
                    "completed": "1",
                    "ext_data": {"download": True, "cardLevel": 1, "canOpen": True},
                }
                biz_data = {"bizType": "common-artifact", "params": params}
                await self._send_callback(OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content="",
                    metadata={"_progress": True, "_data_message": True, "_biz_data": biz_data},
                ))

            # Push the actual video file to the chat channel so the user can watch it.
            # This is done as a normal (non-progress) message so it is always delivered.
            if self._send_callback and self._channel and results[-1].get("success"):
                await self._send_callback(OutboundMessage(
                    channel=self._channel,
                    chat_id=self._chat_id,
                    content=f"🎬 {ep_id} 视频已生成",
                    media=[results[-1]["video_path"]],
                ))

        # Clean up session caches when all episodes are done (not filtered)
        if not episode_filter:
            self._script_tool._session_scripts.pop(self._chat_id, None)
            self._images_tool._session_images.pop(self._chat_id, None)
            self._tts_tool._session_audio.pop(self._chat_id, None)
            # Also clear persisted session files on disk
            sessions_base = getattr(self._script_tool, "_sessions_base", None)
            if sessions_base:
                _clear_session_data(sessions_base, self._chat_id)
            logger.debug("Session caches cleaned (memory + disk) for chat_id={}", self._chat_id)

        # Summary
        ok = sum(1 for r in results if r["success"])
        summary = f"视频合成完成：{ok}/{len(results)} 集成功。\n\n"
        for r in results:
            if r["success"]:
                summary += f"**{r['episode_id']}** ✓ → {r['video_path']}\n"
            else:
                summary += f"**{r['episode_id']}** ✗ → {r['error']}\n"
        summary += f"\n视频输出目录：{self._output_dir}"
        return summary


# ---------------------------------------------------------------------------
# Prompt enhancement helpers
# ---------------------------------------------------------------------------

# Appended to EVERY image prompt to prevent T2I from rendering text on images
_ANTI_TEXT_DIRECTIVE = (
    "Absolutely no text, no words, no letters, no numbers, no watermarks, "
    "no subtitles, no captions, no Chinese characters, no Japanese characters "
    "anywhere in the image."
)

# Appended to EVERY image prompt to prevent T2I from switching to photorealistic
_ANTI_PHOTO_DIRECTIVE = (
    "This MUST be a stylized illustration, NOT a photograph. "
    "Do NOT generate photorealistic or real-person imagery. "
    "Maintain consistent manga/comic illustration style throughout. "
    "Drawn, painted, illustrated — never photographic."
)


def _build_prompt_suffix(script_data: dict) -> str:
    """Build a global suffix from script-level style + character visual tags.

    This is appended to every panel prompt so the T2I model maintains
    consistent character appearance and art style across all panels.
    """
    parts: list[str] = []

    style = script_data.get("style", "")
    if style:
        parts.append(f"Art style: {style}. All panels MUST use this exact style.")
    else:
        parts.append("Art style: detailed manga illustration.")

    characters = script_data.get("characters", [])
    for char in characters:
        name = char.get("name", "")
        tags = char.get("visual_tags", [])
        if name and tags:
            parts.append(f"{name}: {', '.join(tags)}.")

    parts.append(_ANTI_PHOTO_DIRECTIVE)
    parts.append(_ANTI_TEXT_DIRECTIVE)
    return " ".join(parts)


def _extract_scene_anchor(episode: dict) -> str:
    """Extract a scene/location anchor from the episode's first panel prompt.

    The first panel typically establishes the setting.  We pull a short
    location clause so subsequent panels stay in the same environment.
    """
    panels = episode.get("panels", [])
    if not panels:
        return ""

    first_prompt = panels[0].get("image_prompt", "")
    if not first_prompt:
        return ""

    import re

    # Broad location/environment keyword list to cover most scene types
    location_keywords = (
        r"(?:warehouse|interior|exterior|indoor|outdoor|room|hall|chamber|"
        r"temple|palace|castle|tower|fortress|garden|courtyard|"
        r"forest|mountain|river|lake|ocean|sea|desert|field|meadow|swamp|"
        r"city|town|street|market|square|bridge|harbor|port|"
        r"cave|mine|tunnel|underground|"
        r"battlefield|arena|colosseum|stadium|"
        r"throne|cliff|village|alley|dungeon|prison|cell|"
        r"building|rooftop|balcony|terrace|staircase|corridor|"
        r"school|hospital|library|church|shrine|monastery|"
        r"bar|tavern|restaurant|shop|store|office|lab|"
        r"spaceship|station|cockpit|deck|bridge|"
        r"rain|storm|snow|fog|night|dawn|dusk|sunset|sunrise|"
        r"dark|abandoned|ruined|ancient|modern|futuristic|"
        r"bedroom|kitchen|bathroom|basement|attic|garage|"
        r"highway|road|path|trail|woods|jungle|volcano)"
    )
    match = re.search(
        rf"[^.;!]*{location_keywords}[^.;!]*",
        first_prompt,
        re.IGNORECASE,
    )
    if match:
        anchor = match.group(0).strip().rstrip(",").strip()
        if len(anchor) > 150:
            anchor = anchor[:150]
        return f"Consistent scene setting: {anchor}."
    return ""


def _enhance_image_prompt(
    original: str,
    scene_anchor: str,
    global_suffix: str,
) -> str:
    """Enhance a single panel's image prompt for consistency.

    Prepends scene anchor (if not already implied) and appends global suffix
    with character tags + anti-text directive.
    """
    parts = [original.rstrip(". ")]

    if scene_anchor:
        parts.append(scene_anchor)

    if global_suffix:
        parts.append(global_suffix)

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Multi-voice TTS helpers
# ---------------------------------------------------------------------------

# Default Edge TTS voice palette for different speaker roles.
# Keys are normalised role identifiers; values are Edge TTS voice names.
_DEFAULT_VOICE_PALETTE: dict[str, str] = {
    "narrator":   "zh-CN-YunyangNeural",    # male, authoritative news-anchor
    "male1":      "zh-CN-YunxiNeural",       # young male protagonist
    "male2":      "zh-CN-YunjianNeural",     # mature/deep male antagonist
    "female1":    "zh-CN-XiaomoNeural",      # soft female
    "female2":    "zh-CN-XiaoxiaoNeural",    # standard female
    "system":     "zh-CN-XiaoxiaoNeural",    # neutral/robotic
}


def _build_voice_map(script_data: dict) -> dict[str, str]:
    """Build a speaker-name → TTS-voice mapping from the storyboard.

    Strategy:
    1. If the script JSON has ``voice_map``, use it directly.
    2. Otherwise auto-assign from ``characters`` list using the default palette.
    3. Common prefixes (旁白, 系统音, 音效) get fixed voices.
    """
    # Honour explicit mapping if present
    explicit = script_data.get("voice_map")
    if isinstance(explicit, dict) and explicit:
        return explicit

    voice_map: dict[str, str] = {
        "旁白":   _DEFAULT_VOICE_PALETTE["narrator"],
        "系统音":  _DEFAULT_VOICE_PALETTE["system"],
    }

    male_voices = [_DEFAULT_VOICE_PALETTE["male1"], _DEFAULT_VOICE_PALETTE["male2"]]
    female_voices = [_DEFAULT_VOICE_PALETTE["female1"], _DEFAULT_VOICE_PALETTE["female2"]]
    mi, fi = 0, 0  # round-robin indices

    for char in script_data.get("characters", []):
        name = char.get("name", "")
        if not name or name in voice_map:
            continue

        # Detect gender: prefer explicit field, fallback to keywords
        gender = char.get("gender", "").lower()
        if gender in ("female", "f"):
            is_female = True
        elif gender in ("male", "m"):
            is_female = False
        else:
            desc = char.get("description", "").lower()
            tags_str = " ".join(char.get("visual_tags", [])).lower()
            combined = f"{desc} {tags_str}"
            is_female = any(kw in combined for kw in (
                "female", "woman", "girl", "她", "女", "美", "lady",
                "dress", "裙", "姐", "妹",
            ))

        if is_female:
            voice_map[name] = female_voices[fi % len(female_voices)]
            fi += 1
        else:
            voice_map[name] = male_voices[mi % len(male_voices)]
            mi += 1

    return voice_map


def _detect_speaker(dialogue: str) -> tuple[str, str]:
    """Parse a dialogue line to extract (speaker_name, clean_text).

    Handles formats like:
      ``林夜(冰冷)：台词``
      ``旁白(低沉)：台词``
      ``楚天阔：台词``
      ``音效：扑通……``
      ``系统音(机械)：叮！``

    Returns (speaker, text) where speaker may be empty if not detected.
    """
    speaker, _, text = _detect_speaker_emotion(dialogue)
    return speaker, text


def _detect_speaker_emotion(dialogue: str) -> tuple[str, str, str]:
    """Parse a dialogue line to extract (speaker_name, emotion, clean_text).

    Handles formats like:
      ``林夜(冰冷)：台词``   → ("林夜", "冰冷", "台词")
      ``旁白(低沉)：台词``   → ("旁白", "低沉", "台词")
      ``楚天阔：台词``       → ("楚天阔", "", "台词")

    Returns (speaker, emotion, text).
    """
    import re

    m = re.match(
        r"^([^：:（(]{1,15})(?:[（(]([^）)]*)[）)])?[：:](.*)$",
        dialogue.strip(),
        re.DOTALL,
    )
    if m:
        return m.group(1).strip(), (m.group(2) or "").strip(), m.group(3).strip()
    return "", "", dialogue.strip()


def _infer_emotion(speaker: str, text: str) -> str | None:
    """Infer TTS emotion from speaker role + text content when no explicit tag.

    Used as a fallback when the dialogue has no emotion annotation.
    Returns an emotion key matching _EMOTION_PROSODY in tts.py, or None for default.
    """
    has_exclaim = "！" in text or "!" in text
    has_hesitation = "……" in text or "…" in text
    has_question = "？" in text or "?" in text
    has_dash = "——" in text

    if speaker in ("林夜",):
        # Protagonist is cold and composed; exclamations still stay flat/icy
        if has_exclaim and not has_hesitation:
            return "冰冷"
        if has_hesitation:
            return "低沉"
        return "平静"

    elif speaker in ("楚天阔",):
        # Main antagonist: arrogant by default, fearful when hesitant
        if has_hesitation or has_dash:
            return "惊恐"
        if has_exclaim:
            return "嚣张"
        return "嚣张"

    elif speaker in ("黑衣人队长", "黑衣人"):
        return "强硬"

    elif speaker in ("柳如烟",):
        if has_exclaim and has_hesitation:
            return "哭泣"
        if has_question:
            return "惊恐"
        return "悲伤"

    elif speaker in ("系统音",):
        return "机械"

    elif speaker in ("梁天成",):
        if has_hesitation:
            return "强硬"
        if has_exclaim:
            return "嚣张"
        return "轻蔑"

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_audio_duration(path: str) -> float:
    """Return audio duration in seconds via ffprobe. Returns 0.0 on failure."""
    import asyncio
    import json as _json

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        data = _json.loads(stdout)
        return float(data["streams"][0]["duration"])
    except Exception:
        return 0.0


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output using json_repair."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.index("\n") if "\n" in text else 3
        text = text[first_nl + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try json_repair first (handles trailing commas, missing quotes, etc.)
    try:
        result = json_repair.loads(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # Fallback: extract JSON object substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json_repair.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    logger.warning("Failed to parse JSON from LLM output ({} chars), returning raw", len(text))
    return {"raw": text}


def _file_size_str(p: Path) -> str:
    """Human-readable file size, e.g. '1.2 MB'."""
    try:
        size = p.stat().st_size
    except OSError:
        return "??"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


async def _download_image(url: str, dest: Path) -> Path:
    """Download an image URL (or decode base64 data-URL) to a local file.

    Also handles local file paths: if *url* points to an existing file on disk,
    the file is copied to *dest* (with the original extension preserved).
    """
    if url.startswith("data:image"):
        # data:image/png;base64,xxxx
        header, b64data = url.split(",", 1)
        ext = "jpg" if ("jpeg" in header or "jpg" in header) else "png"
        out = dest.with_suffix(f".{ext}")
        out.write_bytes(base64.b64decode(b64data))
        return out

    # Check if this is already a local file path (saved in Phase 2)
    src_path = Path(url)
    if src_path.is_file():
        ext = src_path.suffix or ".png"
        out = dest.with_suffix(ext)
        shutil.copy2(src_path, out)
        return out

    # HTTP/HTTPS URL — download it
    import httpx
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/png")
        ext = "jpg" if "jpeg" in content_type else "png"
        out = dest.with_suffix(f".{ext}")
        out.write_bytes(resp.content)
        return out


def _check_ffmpeg() -> None:
    """Raise early if ffmpeg is not on PATH."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Please install ffmpeg and ensure it is "
            "accessible, or configure tools.exec.pathAppend in config.json."
        )


async def _compose_episode_video(
    episode_id: str,
    panels: list[dict],
    images: dict[str, str],
    audio_map: dict[str, str],
    output_dir: Path,
    aspect_ratio: str = "9:16",
) -> str:
    """Use ffmpeg to compose a single episode video from panel images + audio.

    Strategy:
    1. Download panel images to temp files.
    2. Create ffmpeg concat demuxer file for image slideshow.
    3. Concatenate audio files (or use silence for panels without audio).
    4. Merge slideshow + audio into MP4.

    Temp files are cleaned up after the final output is written.
    """
    _check_ffmpeg()

    # Place the working directory inside the workspace (sibling of output_dir)
    # so that all intermediate files stay within the nanobot workspace.
    work_dir = output_dir.parent / "work" / f"comic_{episode_id}_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        return await _compose_episode_video_inner(
            episode_id, panels, images, audio_map, output_dir, aspect_ratio, work_dir,
        )
    finally:
        # Clean up intermediate working directory after video is written
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            logger.debug("Failed to clean up work dir: {}", work_dir)


async def _compose_episode_video_inner(
    episode_id: str,
    panels: list[dict],
    images: dict[str, str],
    audio_map: dict[str, str],
    output_dir: Path,
    aspect_ratio: str,
    work_dir: Path,
) -> str:
    """Inner implementation of video composition (called within temp-dir context)."""
    # Determine resolution from aspect ratio
    _ASPECT_MAP = {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "4:3": (1440, 1080),
        "3:4": (1080, 1440),
        "1:1": (1080, 1080),
        "21:9": (2560, 1080),
    }
    width, height = _ASPECT_MAP.get(aspect_ratio, (1080, 1920))
    logger.debug("Video resolution: {}x{} (aspect_ratio={})", width, height, aspect_ratio)

    # Step 1: Prepare panel images and durations
    panel_entries: list[dict] = []  # [{image_path, audio_path, duration}]

    for panel in panels:
        panel_id = panel.get("panel_id", "")
        duration = panel.get("duration_s", 3.0)
        image_url = images.get(panel_id, "")

        if not image_url:
            logger.warning("No image for panel {}, skipping", panel_id)
            continue

        # Download image
        img_dest = work_dir / f"{panel_id}"
        try:
            img_path = await _download_image(image_url, img_dest)
            logger.info("[Video] Image ready for {}: {} ({})", panel_id, img_path, _file_size_str(img_path))
        except Exception as e:
            logger.warning("[Video] Failed to download image for {}: {}", panel_id, e)
            continue

        audio_path = audio_map.get(panel_id, "")

        panel_entries.append({
            "panel_id": panel_id,
            "image_path": str(img_path),
            "audio_path": audio_path,
            "duration": duration,
            "dialogue": panel.get("dialogue", ""),
            "narration": panel.get("narration", ""),
        })

    if not panel_entries:
        raise ValueError(f"No valid panels with images found for {episode_id}")
    logger.info("[Video] {} panels ready for composition ({})", len(panel_entries), episode_id)

    # Step 2: Create concat file for images
    concat_file = work_dir / "panels.txt"
    lines = []
    for entry in panel_entries:
        img = entry["image_path"].replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{img}'")
        lines.append(f"duration {entry['duration']}")
    # Repeat last image to avoid ffmpeg cutting it short
    last_img = panel_entries[-1]["image_path"].replace("\\", "/").replace("'", "'\\''")
    lines.append(f"file '{last_img}'")
    concat_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[Video] Concat file written: {}", concat_file)

    # Step 2.5: Generate ASS subtitle file from dialogue/narration
    ass_file = _generate_ass_subtitles(panel_entries, work_dir / "subtitles.ass", width, height)
    if ass_file:
        logger.info("[Video] ASS subtitles: {} ({})", ass_file, _file_size_str(ass_file))
    else:
        logger.info("[Video] No dialogue subtitles to burn in")

    # Step 3: Create slideshow video (with burned-in subtitles)
    slideshow_path = work_dir / "slideshow.mp4"
    vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    if ass_file:
        # ffmpeg subtitles filter needs forward-slash paths; colons escaped on Windows
        ass_path_escaped = str(ass_file).replace("\\", "/").replace(":", "\\:")
        vf += f",subtitles='{ass_path_escaped}'"
    slideshow_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-vf", vf,
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "fast",
        str(slideshow_path),
    ]

    logger.info("[Video] Running ffmpeg slideshow: {} -> {}", concat_file.name, slideshow_path.name)
    proc = await asyncio.create_subprocess_exec(
        *slideshow_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error("[Video] ffmpeg slideshow FAILED (rc={}): {}", proc.returncode, stderr.decode()[-500:])
        raise RuntimeError(f"ffmpeg slideshow failed: {stderr.decode()[-500:]}")
    logger.info("[Video] Slideshow created: {} ({})", slideshow_path, _file_size_str(slideshow_path))

    # Step 4: Build per-panel audio (normalize all to same format for concat)
    audio_entries = [e for e in panel_entries if e["audio_path"]]
    final_output = output_dir / f"{episode_id}.mp4"

    if audio_entries:
        # Normalize each panel's audio to consistent format AND align to panel duration.
        # - apad=whole_dur pads silence if TTS is shorter than panel duration
        # - -t truncates if TTS is longer
        # This ensures each audio clip is exactly duration_s, so concat aligns with slideshow.
        normalized_audio_paths: list[str] = []
        for entry in panel_entries:
            dur = entry["duration"]
            if entry["audio_path"]:
                norm_path = work_dir / f"norm_{entry['panel_id']}.mp3"
                norm_cmd = [
                    "ffmpeg", "-y",
                    "-i", entry["audio_path"],
                    "-ar", "44100", "-ac", "1",
                    "-af", f"apad=whole_dur={dur}",
                    "-t", str(dur),
                    "-c:a", "libmp3lame", "-b:a", "128k",
                    str(norm_path),
                ]
                sp = await asyncio.create_subprocess_exec(
                    *norm_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await sp.communicate()
                if sp.returncode == 0:
                    normalized_audio_paths.append(str(norm_path))
                else:
                    logger.warning("Audio normalize failed for {}, using silence", entry["panel_id"])
                    sil_path = work_dir / f"silence_{entry['panel_id']}.mp3"
                    await _generate_silence(sil_path, dur)
                    normalized_audio_paths.append(str(sil_path))
            else:
                silence_path = work_dir / f"silence_{entry['panel_id']}.mp3"
                await _generate_silence(silence_path, dur)
                normalized_audio_paths.append(str(silence_path))

        audio_concat_file = work_dir / "audio_list.txt"
        audio_lines = []
        for ap in normalized_audio_paths:
            ap_escaped = ap.replace("\\", "/").replace("'", "'\\''")
            audio_lines.append(f"file '{ap_escaped}'")
        audio_concat_file.write_text("\n".join(audio_lines), encoding="utf-8")

        logger.info("[Video] Normalized {} audio clips, merging...", len(normalized_audio_paths))
        merged_audio = work_dir / "merged_audio.mp3"
        audio_merge_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(audio_concat_file),
            "-c", "copy",
            str(merged_audio),
        ]
        proc = await asyncio.create_subprocess_exec(
            *audio_merge_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("[Video] Audio concat FAILED, producing video without audio: {}", stderr.decode()[-300:])
            shutil.copy2(str(slideshow_path), str(final_output))
            logger.info("[Video] Final output (no audio): {} ({})", final_output, _file_size_str(final_output))
            return str(final_output)
        logger.info("[Video] Merged audio: {} ({})", merged_audio, _file_size_str(merged_audio))

        # Step 5: Merge video + audio
        logger.info("[Video] Merging video + audio -> {}", final_output)
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", str(slideshow_path),
            "-i", str(merged_audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(final_output),
        ]
        proc = await asyncio.create_subprocess_exec(
            *merge_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("[Video] ffmpeg merge FAILED (rc={}): {}", proc.returncode, stderr.decode()[-500:])
            raise RuntimeError(f"ffmpeg merge failed: {stderr.decode()[-500:]}")
    else:
        # No audio — just copy slideshow as final output
        shutil.copy2(str(slideshow_path), str(final_output))
        logger.info("[Video] No audio entries, copied slideshow as final output")

    logger.info("[Video] Final video: {} ({})", final_output, _file_size_str(final_output))
    return str(final_output)


async def _generate_silence(path: Path, duration: float) -> None:
    """Generate a silent MP3 file with consistent parameters."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration),
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(path),
    ]
    sp = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await sp.communicate()


def _generate_ass_subtitles(
    panel_entries: list[dict],
    output_path: Path,
    width: int,
    height: int,
) -> Path | None:
    """Generate an ASS subtitle file from panel dialogue only.

    Narration is already played as TTS audio and should NOT appear as on-screen
    text.  Only character dialogue is burned into the video as subtitles.

    Returns the output path if subtitles were written, or None if no dialogue found.
    """
    import platform as _platform

    # Collect subtitle events with cumulative timing — dialogue only
    events: list[dict] = []
    t = 0.0  # cumulative start time in seconds
    for entry in panel_entries:
        dur = entry.get("duration", 3.0)
        dialogue = (entry.get("dialogue") or "").strip()

        if dialogue:
            events.append({
                "start": t, "end": t + dur,
                "style": "Dialogue", "text": dialogue,
            })
        t += dur

    if not events:
        return None

    # Font selection: prefer Microsoft YaHei on Windows, fallback to sans-serif
    if _platform.system() == "Windows":
        font_name = "Microsoft YaHei"
    else:
        font_name = "Noto Sans CJK SC"

    # Adaptive font size: ~18 CJK chars per line on 1080px wide video
    base_size = max(32, int(height / 40))
    margin_h = int(width * 0.06)  # 6% horizontal margin each side

    # Build ASS file content
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Dialogue: white text, black outline+shadow, bottom-center (Alignment=2)
        f"Style: Dialogue,{font_name},{base_size},"
        "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,3,1,2,{margin_h},{margin_h},{int(height * 0.08)},1\n"
    )

    lines = [header, "[Events]",
             "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    for ev in events:
        start_ts = _seconds_to_ass_time(ev["start"])
        end_ts = _seconds_to_ass_time(ev["end"])
        # Escape special ASS chars and convert newlines
        text = ev["text"].replace("\\", "\\\\").replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{start_ts},{end_ts},{ev['style']},,0,0,0,,{text}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8-sig")
    logger.info("ASS subtitles generated: {} events -> {}", len(events), output_path)
    return output_path


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    cs = int((s - int(s)) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"
