"""HuggingFace Inference API provider — free tier available."""

import os
import time
import httpx
from providers.base import BaseLLMProvider, LLMResponse


class HuggingFaceProvider(BaseLLMProvider):
    """
    Calls HuggingFace Inference API (free tier).
    Set HF_API_KEY in your .env file.
    Default model: mistralai/Mistral-7B-Instruct-v0.1
    """

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.model = model
        self.api_key = os.getenv("HF_API_KEY")
        if not self.api_key:
            raise EnvironmentError("HF_API_KEY not set. Add it to your .env file.")

    def generate(self, prompt: str) -> LLMResponse:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.7,
                "return_full_text": False,
            },
        }

        start = time.perf_counter()
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = resp.json()
        if isinstance(data, list):
            text = data[0].get("generated_text", "").strip()
        else:
            text = str(data)

        return LLMResponse(
            text=text,
            latency_ms=latency_ms,
            provider="huggingface",
            model=self.model,
        )
