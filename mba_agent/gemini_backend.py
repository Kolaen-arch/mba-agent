"""
Gemini (Google) backend implementation.
Uses the google-genai SDK for Gemini 3.x models with Google Search grounding.
"""

from typing import Generator

from .llm_adapter import LLMBackend

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GeminiBackend(LLMBackend):
    """Gemini backend using the google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-3.1-pro-preview", search: bool = True):
        if not HAS_GEMINI:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.search = search  # Enable Google Search grounding by default
        self.cache_name = None  # Set after context upload

    def _build_config(
        self, system: str, thinking_budget: int, max_output_tokens: int,
        search: bool | None = None,
    ) -> "types.GenerateContentConfig":
        """Build GenerateContentConfig with optional thinking and search."""
        config_kwargs = {
            "system_instruction": system,
            "max_output_tokens": max_output_tokens,
        }

        if thinking_budget > 0:
            # Gemini 3.x uses thinking_level (str), Gemini 2.5 uses thinking_budget (int)
            if "gemini-3" in self.model:
                # Map budget to level: high (default), medium, low
                if thinking_budget >= 8000:
                    level = "high"
                elif thinking_budget >= 4000:
                    level = "medium"
                else:
                    level = "low"
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            else:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=True,
                )
        elif "gemini-3" in self.model:
            # Gemini 3.x: always enable thinking at high by default
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="high")

        # Google Search grounding
        use_search = search if search is not None else self.search
        if use_search:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        return types.GenerateContentConfig(**config_kwargs)

    def _build_contents(
        self, user_message: str, history: list[dict] | None,
    ) -> list:
        """Build contents list from history + current message."""
        contents = []
        if history:
            for msg in history:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        ))
        return contents

    def _extract_usage(self, response) -> dict:
        """Extract token usage from a Gemini response."""
        usage = {}
        meta = getattr(response, 'usage_metadata', None)
        if meta:
            usage = {
                "input_tokens": getattr(meta, 'prompt_token_count', 0) or 0,
                "output_tokens": getattr(meta, 'candidates_token_count', 0) or 0,
                "cache_read_input_tokens": getattr(meta, 'cached_content_token_count', 0) or 0,
                "cache_creation_input_tokens": 0,
            }
        return usage

    @staticmethod
    def _extract_grounding(response) -> list[dict]:
        """Extract grounding sources from a Gemini response with Google Search."""
        sources = []
        meta = getattr(response, 'candidates', [])
        if not meta:
            return sources
        grounding = getattr(meta[0], 'grounding_metadata', None)
        if not grounding:
            return sources
        for chunk in getattr(grounding, 'grounding_chunks', []) or []:
            web = getattr(chunk, 'web', None)
            if web:
                sources.append({
                    "title": getattr(web, 'title', '') or '',
                    "uri": getattr(web, 'uri', '') or '',
                })
        return sources

    def call(
        self,
        system: str,
        user_message: str,
        mode: str = "chat",
        thinking_budget: int = 0,
        max_output_tokens: int = 16000,
        history: list[dict] | None = None,
    ) -> str:
        config = self._build_config(system, thinking_budget, max_output_tokens)
        contents = self._build_contents(user_message, history)

        kwargs = {"model": self.model, "contents": contents, "config": config}
        if self.cache_name:
            kwargs["cached_content"] = self.cache_name

        response = self.client.models.generate_content(**kwargs)

        # Extract text (skip thought parts)
        text = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if not getattr(part, 'thought', False):
                    text += part.text or ""

        # Append grounding sources if available
        sources = self._extract_grounding(response)
        if sources:
            text += "\n\n---\n**Sources:**\n"
            for s in sources:
                if s["uri"]:
                    text += f"- [{s['title']}]({s['uri']})\n"

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
        config = self._build_config(system, thinking_budget, max_output_tokens)
        contents = self._build_contents(user_message, history)

        kwargs = {"model": self.model, "contents": contents, "config": config}
        if self.cache_name:
            kwargs["cached_content"] = self.cache_name

        yield {"type": "model", "model": self.model, "thinking_enabled": thinking_budget > 0}

        full_text = ""
        in_thinking = False
        usage = {}
        last_response = None

        try:
            response_stream = self.client.models.generate_content_stream(**kwargs)
            for chunk in response_stream:
                last_response = chunk
                # Extract usage from each chunk (last one has final counts)
                usage = self._extract_usage(chunk) or usage

                if not chunk.candidates:
                    continue

                for part in chunk.candidates[0].content.parts:
                    part_text = part.text or ""
                    is_thought = getattr(part, 'thought', False)

                    if is_thought:
                        if not in_thinking:
                            in_thinking = True
                            yield {"type": "thinking_start"}
                        yield {"type": "thinking", "text": part_text}
                    else:
                        if in_thinking:
                            in_thinking = False
                            yield {"type": "thinking_done"}
                        full_text += part_text
                        yield {"type": "text", "text": part_text}

            if in_thinking:
                yield {"type": "thinking_done"}

            # Append grounding sources at end of stream
            if last_response:
                sources = self._extract_grounding(last_response)
                if sources:
                    src_text = "\n\n---\n**Sources:**\n"
                    for s in sources:
                        if s["uri"]:
                            src_text += f"- [{s['title']}]({s['uri']})\n"
                    full_text += src_text
                    yield {"type": "text", "text": src_text}

        except GeneratorExit:
            return

        yield {"type": "done", "full_text": full_text, "model": self.model, "usage": usage}

    def get_model_name(self) -> str:
        return self.model
