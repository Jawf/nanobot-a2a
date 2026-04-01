"""Text-to-Speech service.

Supports:
  - Edge TTS (free, default) via the ``edge-tts`` package
  - OpenAI TTS via the standard API

Edge TTS emotion constraints (free endpoint)
---------------------------------------------
The free endpoint (speech.platform.bing.com) ONLY supports:
  • <prosody rate pitch volume>  — via Communicate(rate=, pitch=, volume=)

<break>, <emphasis>, <mstts:express-as> are all rejected (no audio returned).

Strategy:
  1. Strong prosody values per emotion (monkey-patch mkssml so volume applies)
  2. Text preprocessing — inject Chinese punctuation to control rhythm/pauses:
       ，  → short pause (~150ms)   for hesitation / deliberate pacing
       。  → full stop pause        for solemn endings
       ……  → edge-tts natural hesitation rendering
     For 激动/愤怒: remove soft pauses, add ！ emphasis markers
     For 冰冷/淡然: insert ， between clauses for cold deliberate rhythm
     For 哭泣/悲伤: insert ，after emotional words for sob-like breaks
     For 破音颤抖: split on ……/——/？ with explicit ，to simulate stuttering
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Emotion → (rate, pitch, volume)
# ---------------------------------------------------------------------------
_EMOTION_PROSODY: dict[str, tuple[str, str, str]] = {
    # ── Calm / Narration ─────────────────────────────────────────────────
    "低沉":     ("-25%", "-10Hz", "+0%"),
    "沉稳":     ("-20%", "-8Hz",  "+0%"),
    "威严":     ("-22%", "-12Hz", "+8%"),
    "空灵":     ("-15%", "+6Hz",  "-15%"),
    # ── Cold / Detached ──────────────────────────────────────────────────
    "冰冷":     ("-22%", "-8Hz",  "-5%"),
    "平静":     ("-15%", "+0Hz",  "+0%"),
    "淡然":     ("-20%", "-5Hz",  "-5%"),
    # ── Anger / Intensity ────────────────────────────────────────────────
    "愤怒":     ("+35%", "+12Hz", "+25%"),
    "嚣张":     ("+30%", "+10Hz", "+20%"),
    "强硬":     ("-5%",  "-3Hz",  "+18%"),
    "激动":     ("+40%", "+15Hz", "+30%"),
    "嘶吼":     ("+40%", "+15Hz", "+30%"),
    # ── Fear / Trembling ─────────────────────────────────────────────────
    "惊恐":     ("+30%", "+12Hz", "+10%"),
    "破音颤抖": ("+18%", "+10Hz", "+8%"),
    "紧张":     ("+18%", "+6Hz",  "+5%"),
    # ── Sadness / Crying ─────────────────────────────────────────────────
    "哭泣":     ("-18%", "-5Hz",  "-8%"),
    "悲伤":     ("-22%", "-8Hz",  "-12%"),
    "抽泣":     ("-15%", "-4Hz",  "-8%"),
    # ── Contempt / Mockery ───────────────────────────────────────────────
    "轻蔑":     ("-10%", "+4Hz",  "+0%"),
    "阴笑":     ("-15%", "-5Hz",  "+0%"),
    "狂笑":     ("+25%", "+8Hz",  "+20%"),
    "嘲讽":     ("-8%",  "+5Hz",  "+0%"),
    # ── Mechanical / System ──────────────────────────────────────────────
    "机械":     ("-12%", "-20Hz", "-15%"),
    # ── Other ────────────────────────────────────────────────────────────
    "宣告":     ("-12%", "-6Hz",  "+10%"),
    "低语":     ("-28%", "+0Hz",  "-25%"),
    "轻松":     ("+5%",  "+3Hz",  "+0%"),
}

_DEFAULT_PROSODY = ("+0%", "+0Hz", "+0%")


def _preprocess_text(text: str, emotion: str | None) -> str:
    """Manipulate Chinese punctuation to improve rhythm and emotion.

    Chinese TTS engines render punctuation as natural pauses/inflections,
    so inserting / replacing punctuation shapes emotional cadence.
    """
    t = text.strip()

    if emotion in ("激动", "愤怒", "嘶吼", "嚣张", "狂笑"):
        # High energy: compress hesitations, ensure exclamation marks
        t = t.replace("……", "").replace("…", "")
        t = t.replace("——", "，")
        if t and t[-1] not in "！!？?。":
            t += "！"

    elif emotion in ("冰冷", "淡然", "平静", "威严"):
        # Cold/deliberate: insert pauses between clauses for measured delivery
        t = re.sub(r'([^，。！？…—\s]{3,6})([^，。！？…—\s])', r'\1，\2', t, count=3)
        # Remove exclamation, replace with period for flat tone
        t = t.replace("！", "。")

    elif emotion in ("破音颤抖", "惊恐", "紧张"):
        # Trembling: split on existing pauses to simulate stuttering
        t = t.replace("……", "，，").replace("——", "，，")
        # Add comma after every 2-3 chars for short stutter effect
        t = re.sub(r'([^，。！？\s]{2,3})([^，。！？\s])', r'\1，\2', t, count=4)

    elif emotion in ("哭泣", "悲伤", "抽泣"):
        # Sobbing: slow down with extra pauses, keep question/exclamation
        t = t.replace("……", "，……").replace("——", "，——")
        t = re.sub(r'([^，。！？\s]{4,})([！？])', r'\1，\2', t)

    elif emotion in ("低沉", "沉稳", "威严", "机械"):
        # Deep/solemn: convert exclamations to periods for flat gravitas
        t = t.replace("！", "。")
        t = t.replace("……", "，")

    return t


class TTSService:
    """Async TTS client that writes audio files."""

    def __init__(
        self,
        provider: str = "edge",
        api_key: str = "",
        api_base: str = "",
        model: str = "",
        voice: str = "zh-CN-XiaoxiaoNeural",
        timeout: int = 60,
        output_dir: Path | None = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.voice = voice
        self.timeout = timeout
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "nanobot_tts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(
        self,
        text: str,
        filename: str,
        voice: str | None = None,
        emotion: str | None = None,
    ) -> str:
        """Generate speech audio and return the file path."""
        if not text or not text.strip():
            raise ValueError("TTS text must not be empty")

        output_path = self.output_dir / f"{filename}.mp3"
        use_voice = voice or self.voice

        if self.provider == "edge":
            return await self._synthesize_edge(text, output_path, use_voice, emotion)
        elif self.provider == "openai":
            return await self._synthesize_openai(text, output_path, use_voice)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

    async def _synthesize_edge(
        self,
        text: str,
        output_path: Path,
        voice: str,
        emotion: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Edge TTS with prosody and text-level emotion shaping.

        Volume is injected via monkey-patching mkssml (volume= param not
        supported in the Communicate constructor directly in older versions).
        Text is preprocessed to shape rhythm via Chinese punctuation.
        """
        import edge_tts
        import edge_tts.communicate as _et_comm

        rate, pitch, volume = _EMOTION_PROSODY.get(emotion or "", _DEFAULT_PROSODY)
        processed_text = _preprocess_text(text, emotion)

        original_mkssml = _et_comm.mkssml

        def _volume_mkssml(tc, escaped_text):
            if isinstance(escaped_text, bytes):
                escaped_text = escaped_text.decode("utf-8")
            return (
                "<speak version='1.0'"
                " xmlns='http://www.w3.org/2001/10/synthesis'"
                " xml:lang='zh-CN'>"
                f"<voice name='{tc.voice}'>"
                f"<prosody rate='{tc.rate}' pitch='{tc.pitch}' volume='{volume}'>"
                f"{escaped_text}"
                "</prosody>"
                "</voice>"
                "</speak>"
            )

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            _et_comm.mkssml = _volume_mkssml
            try:
                communicate = edge_tts.Communicate(
                    processed_text, voice, rate=rate, pitch=pitch,
                )
                await communicate.save(str(output_path))
                logger.info(
                    "Edge TTS: {} ({} chars→{}, emotion={}, rate={} pitch={} vol={})",
                    output_path.name, len(text), len(processed_text),
                    emotion or "none", rate, pitch, volume,
                )
                return str(output_path)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = retry_delay * attempt
                    logger.warning(
                        "Edge TTS attempt {}/{} failed for {} ({}), retrying in {:.0f}s",
                        attempt, max_retries, output_path.name, exc, wait,
                    )
                    await asyncio.sleep(wait)
            finally:
                _et_comm.mkssml = original_mkssml

        raise RuntimeError(
            f"Edge TTS failed after {max_retries} attempts: {last_exc}"
        ) from last_exc

    async def _synthesize_openai(self, text: str, output_path: Path, voice: str) -> str:
        """Use OpenAI TTS API."""
        import httpx

        url = (
            self.api_base.rstrip("/") if self.api_base else "https://api.openai.com/v1"
        ) + "/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model or "tts-1",
            "input": text,
            "voice": voice or "alloy",
            "response_format": "mp3",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
        logger.info("OpenAI TTS: {} ({} chars)", output_path.name, len(text))
        return str(output_path)
