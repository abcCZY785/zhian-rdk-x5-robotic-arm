"""OpenAI-compatible Xiaomi MIMO client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class MimoClientError(RuntimeError):
    """Raised when the Xiaomi MIMO API call fails."""


class MimoClient:
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout: float = 30.0):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("api_key cannot be empty")
        if not self.model:
            raise ValueError("model cannot be empty")

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.1) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MimoClientError(f"http {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise MimoClientError(f"request failed: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise MimoClientError(f"invalid json response: {body[:400]}") from exc

