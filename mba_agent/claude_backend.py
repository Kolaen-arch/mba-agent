"""
Claude (Anthropic) backend implementation.
Extracted from agent.py — handles all Anthropic-specific API interaction.
"""

from typing import Generator

import anthropic

from .llm_adapter import LLMBackend


class ClaudeBackend(LLMBackend):
    """Claude backend using the Anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-6",
        enable_prompt_cache: bool = True,
    ):
        self.client = anthropic.Anthropic(api_key=api_key, timeout=120.0)
        self.model = model
        self.enable_prompt_cache = enable_prompt_cache

    def _build_system_with_cache(self, system: str) -> list[dict] | str:
        """Wrap system prompt in cache_control block for prompt caching."""
        if not self.enable_prompt_cache:
            return system
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def call(
        self,
        system: str,
        user_message: str,
        mode: str = "chat",
        thinking_budget: int = 0,
        max_output_tokens: int = 16000,
        history: list[dict] | None = None,
    ) -> str:
        messages = list(history or []) + [{"role": "user", "content": user_message}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": self._build_system_with_cache(system),
            "messages": messages,
        }

        if thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            kwargs["max_tokens"] = max_output_tokens + thinking_budget

        response = self.client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return text

    def stream(
        self,
        system: str,
        user_message: str,
        mode: str = "chat",
        thinking_budget: int = 0,
        max_output_tokens: int = 16000,
        history: list[dict] | None = None,
    ) -> Generator[dict, None, None]:
        messages = list(history or []) + [{"role": "user", "content": user_message}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "system": self._build_system_with_cache(system),
            "messages": messages,
        }

        if thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            kwargs["max_tokens"] = max_output_tokens + thinking_budget

        yield {"type": "model", "model": self.model, "thinking_enabled": thinking_budget > 0}

        full_text = ""
        is_thinking = False
        usage = {}

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event, 'content_block'):
                                if event.content_block.type == 'thinking':
                                    is_thinking = True
                                    yield {"type": "thinking_start"}
                                elif event.content_block.type == 'text':
                                    is_thinking = False
                        elif event.type == 'content_block_delta':
                            if hasattr(event, 'delta'):
                                if hasattr(event.delta, 'thinking'):
                                    yield {"type": "thinking", "text": event.delta.thinking}
                                elif hasattr(event.delta, 'text'):
                                    full_text += event.delta.text
                                    yield {"type": "text", "text": event.delta.text}
                        elif event.type == 'content_block_stop':
                            if is_thinking:
                                yield {"type": "thinking_done"}
                                is_thinking = False

                # Extract usage from final message
                try:
                    final = stream.get_final_message()
                    if final and hasattr(final, 'usage'):
                        u = final.usage
                        usage = {
                            "input_tokens": getattr(u, 'input_tokens', 0),
                            "output_tokens": getattr(u, 'output_tokens', 0),
                            "cache_read_input_tokens": getattr(u, 'cache_read_input_tokens', 0),
                            "cache_creation_input_tokens": getattr(u, 'cache_creation_input_tokens', 0),
                        }
                except Exception:
                    pass
        except GeneratorExit:
            return  # Client disconnected

        yield {"type": "done", "full_text": full_text, "model": self.model, "usage": usage}

    def get_model_name(self) -> str:
        return self.model
