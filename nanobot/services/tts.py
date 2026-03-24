"""Text-to-Speech service.

Supports:
  - Edge TTS (free, default) via the ``edge-tts`` package
  - OpenAI TTS via the standard API
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from loguru import logger


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
    ) -> str:
        """Generate speech audio and return the file path.

        Parameters
        ----------
        text:
            The text to convert to speech.
        filename:
            Output filename (without extension; .mp3 will be appended).
        voice:
            Override voice for this call (optional).

        Returns
        -------
        str
            Absolute path to the generated audio file.
        """
        if not text or not text.strip():
            raise ValueError("TTS text must not be empty")

        output_path = self.output_dir / f"{filename}.mp3"
        use_voice = voice or self.voice

        if self.provider == "edge":
            return await self._synthesize_edge(text, output_path, use_voice)
        elif self.provider == "openai":
            return await self._synthesize_openai(text, output_path, use_voice)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

    async def _synthesize_edge(self, text: str, output_path: Path, voice: str) -> str:
        """Use edge-tts (free Microsoft Edge TTS)."""
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        logger.info("Edge TTS: generated {} ({} chars)", output_path.name, len(text))
        return str(output_path)

    async def _synthesize_openai(self, text: str, output_path: Path, voice: str) -> str:
        """Use OpenAI TTS API."""
        import httpx

        url = (self.api_base.rstrip("/") if self.api_base else "https://api.openai.com/v1") + "/audio/speech"
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
        logger.info("OpenAI TTS: generated {} ({} chars)", output_path.name, len(text))
        return str(output_path)
