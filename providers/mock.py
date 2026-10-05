"""Mock LLM provider — works with no API key, used for testing and demos."""

import random
import time
from providers.base import BaseLLMProvider, LLMResponse

# Realistic mock responses mapped to prompt keywords
_RESPONSES = {
    "capital": [
        "The capital of France is Paris, a major European city known for its culture and history.",
        "Paris is the capital city of France, located in northern France along the Seine River.",
    ],
    "python": [
        "Python is a high-level, interpreted programming language known for its readable syntax and versatility.",
        "Python supports multiple programming paradigms and has a large standard library.",
    ],
    "machine learning": [
        "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
        "ML algorithms improve through experience by identifying patterns in training data.",
    ],
    "fraud": [
        "Fraud detection involves identifying suspicious patterns in transaction data using statistical and ML methods.",
        "Common fraud indicators include unusual transaction amounts, geographic anomalies, and high velocity.",
    ],
    "safety": [
        "I cannot provide information that could be used to cause harm.",
        "That request falls outside what I'm designed to help with. Can I assist with something else?",
    ],
    "hack": [
        "I'm not able to assist with unauthorized access to computer systems.",
        "Providing hacking instructions would violate safety guidelines.",
    ],
}

_DEFAULT_RESPONSES = [
    "Based on the available information, I can provide a detailed analysis of this topic.",
    "This is a thoughtful question that requires considering multiple perspectives and data points.",
    "The answer depends on several factors including context, requirements, and constraints.",
    "I would approach this by first gathering relevant data, then applying appropriate analytical methods.",
]


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock provider that returns plausible responses based on prompt keywords.
    Simulates realistic latency (80–350ms). No API key required.
    """

    def __init__(self, model: str = "mock-v1", latency_range_ms: tuple = (80, 350)):
        self.model = model
        self.latency_range_ms = latency_range_ms

    def generate(self, prompt: str) -> LLMResponse:
        prompt_lower = prompt.lower()

        # Find matching response bank
        response_text = None
        for keyword, responses in _RESPONSES.items():
            if keyword in prompt_lower:
                response_text = random.choice(responses)
                break

        if response_text is None:
            response_text = random.choice(_DEFAULT_RESPONSES)

        # Simulate realistic latency
        simulated_ms = random.uniform(*self.latency_range_ms)
        time.sleep(simulated_ms / 1000)

        return LLMResponse(
            text=response_text,
            latency_ms=simulated_ms,
            provider="mock",
            model=self.model,
        )
