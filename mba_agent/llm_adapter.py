"""
Abstract LLM backend interface.
Both Claude and Gemini backends implement this.
"""

from abc import ABC, abstractmethod
from typing import Generator


class LLMBackend(ABC):
    """Abstract LLM backend. Claude and Gemini implement this."""

    @abstractmethod
    def call(
        self,
        system: str,
        user_message: str,
        mode: str = "chat",
        thinking_budget: int = 0,
        max_output_tokens: int = 16000,
        history: list[dict] | None = None,
    ) -> str:
        """Non-streaming call. Returns complete text."""
        ...

    @abstractmethod
    def stream(
        self,
        system: str,
        user_message: str,
        mode: str = "chat",
        thinking_budget: int = 0,
        max_output_tokens: int = 16000,
        history: list[dict] | None = None,
    ) -> Generator[dict, None, None]:
        """
        Streaming call. Yields dicts with these event types:
          {"type": "model", "model": "...", "thinking_enabled": bool}
          {"type": "thinking_start"}
          {"type": "thinking", "text": "..."}
          {"type": "thinking_done"}
          {"type": "text", "text": "..."}
          {"type": "done", "full_text": "...", "model": "...", "usage": {...}}
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model string for display."""
        ...
