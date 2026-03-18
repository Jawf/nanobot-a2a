"""Gemini image generation service.

Calls the Gemini text-to-image synthesis API, preserving the same contract
as the workbench-agent ``gemini_service.generate_image`` function.

Endpoint: POST {api_base}/synthesis/t2i
"""

from __future__ import annotations

import httpx
from loguru import logger


class GeminiImageService:
    """Async client for Gemini text-to-image API."""

    def __init__(self, api_base: str, api_key: str, model: str, timeout: int = 120):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
    ) -> str:
        """Generate an image and return its URL (or base64 data-url).

        Parameters
        ----------
        prompt:
            English image description for the text-to-image model.
        aspect_ratio:
            Image aspect ratio, e.g. ``"16:9"``, ``"1:1"``.

        Returns
        -------
        str
            Image URL from the provider, or a ``data:image/png;base64,...``
            data-url if the provider returns raw base64.
        """
        url = f"{self.api_base}/synthesis/t2i"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "sync": True,
            "prompt": prompt,
            "extra_body": {
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                    },
                },
            },
        }

        logger.info("Gemini image: requesting model={}, prompt={}", self.model, prompt[:120])

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)

            # Retry once on 504 (gateway timeout)
            if resp.status_code == 504:
                logger.warning("Gemini image: 504 timeout, retrying once …")
                resp = await client.post(url, json=payload, headers=headers)

            resp.raise_for_status()
            return self._parse_response(resp)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(resp: httpx.Response) -> str:
        """Extract image URL or base64 from provider response."""
        try:
            data = resp.json()
        except Exception:
            # Raw base64 body
            raw = resp.text.strip().strip('"')
            return f"data:image/png;base64,{raw}"

        # Format 1: {"data": [{"url": "https://..."}]}
        if isinstance(data.get("data"), list):
            for item in data["data"]:
                if isinstance(item, dict) and item.get("url"):
                    logger.info("Gemini image: got URL ({} chars)", len(item["url"]))
                    return item["url"]

        # Format 2: {"data": "<base64 string>"}
        raw_data = data.get("data")
        if isinstance(raw_data, str):
            raw_data = raw_data.strip().strip('"')
            if raw_data.startswith("data:image"):
                return raw_data
            return f"data:image/png;base64,{raw_data}"

        # Format 3: {"url": "..."}
        if data.get("url"):
            return data["url"]

        raise ValueError(f"Cannot extract image from Gemini response: {str(data)[:200]}")
