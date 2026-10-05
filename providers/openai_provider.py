"""OpenAI provider — requires OPENAI_API_KEY environment variable."""

import os
import time
import httpx
from providers.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """
    Calls OpenAI Chat Completions API (gpt-3.5-turbo by default).
    Set OPENAI_API_KEY in your .env file to use this provider.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Add it to your .env file.")

    def generate(self, prompt: str) -> LLMResponse:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.7,
        }

        start = time.perf_counter()
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()

        return LLMResponse(
            text=text,
            latency_ms=latency_ms,
            provider="openai",
            model=self.model,
        )
