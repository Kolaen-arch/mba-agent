"""
OpenRouter backend implementation.
Uses the OpenAI-compatible API at https://openrouter.ai/api/v1
Supports models like openai/gpt-5.4, anthropic/claude-*, meta-llama/*, etc.
"""

import logging
from typing import Generator

from openai import OpenAI

from .llm_adapter import LLMBackend

log = logging.getLogger(__name__)


class OpenRouterBackend(LLMBackend):
    """OpenRouter backend using the OpenAI SDK with custom base_url."""

    def __init__(self, api_key: str, model: str = "openai/gpt-5.4"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=180.0,
        )
        self.model = model

    def call(
        self,
        system: str,
        user_message: str,
        mode: str = "chat",
        thinking_budget: int = 0,
        max_output_tokens: int = 16000,
        history: list[dict] | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_output_tokens,
        }

        # OpenRouter passes reasoning_effort for models that support it
        if thinking_budget > 0:
            if thinking_budget >= 8000:
                kwargs["extra_body"] = {"reasoning": {"effort": "high"}}
            elif thinking_budget >= 4000:
                kwargs["extra_body"] = {"reasoning": {"effort": "medium"}}
            else:
                kwargs["extra_body"] = {"reasoning": {"effort": "low"}}

        response = self.client.chat.completions.create(**kwargs)

        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""
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
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if thinking_budget > 0:
            if thinking_budget >= 8000:
                kwargs["extra_body"] = {"reasoning": {"effort": "high"}}
            elif thinking_budget >= 4000:
                kwargs["extra_body"] = {"reasoning": {"effort": "medium"}}
            else:
                kwargs["extra_body"] = {"reasoning": {"effort": "low"}}

        yield {"type": "model", "model": self.model, "thinking_enabled": thinking_budget > 0}

        full_text = ""
        in_reasoning = False
        usage = {}

        try:
            response_stream = self.client.chat.completions.create(**kwargs)

            for chunk in response_stream:
                # Usage info comes in the final chunk
                if chunk.usage:
                    usage = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Some OpenRouter models return reasoning_content
                reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                if reasoning:
                    if not in_reasoning:
                        in_reasoning = True
                        yield {"type": "thinking_start"}
                    yield {"type": "thinking", "text": reasoning}
                    continue

                content = delta.content
                if content:
                    if in_reasoning:
                        in_reasoning = False
                        yield {"type": "thinking_done"}
                    full_text += content
                    yield {"type": "text", "text": content}

            if in_reasoning:
                yield {"type": "thinking_done"}

        except GeneratorExit:
            return

        yield {"type": "done", "full_text": full_text, "model": self.model, "usage": usage}

    def get_model_name(self) -> str:
        return self.model
