"""Abstract base class for LLM providers."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    latency_ms: float
    provider: str
    model: str


class BaseLLMProvider(ABC):
    """Every provider must implement this interface."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Send prompt, return response with timing."""
        ...

    def _timed_call(self, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
