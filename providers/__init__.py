"""Provider registry — maps provider name strings to classes."""

from providers.mock import MockLLMProvider
from providers.openai_provider import OpenAIProvider
from providers.huggingface_provider import HuggingFaceProvider


def get_provider(name: str):
    """Return an instantiated provider by name. Falls back to mock on missing API keys."""
    try:
        if name == "openai":
            return OpenAIProvider()
        elif name == "huggingface":
            return HuggingFaceProvider()
        else:
            return MockLLMProvider()
    except EnvironmentError as e:
        print(f"[WARNING] {e} — falling back to mock provider.")
        return MockLLMProvider()
