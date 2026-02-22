"""
Agent: Claude API interaction layer.
Supports streaming, extended thinking, model routing per task,
and token budget management.
"""

import json
from typing import Generator

import anthropic

from . import prompts


# Rough token estimation: 1 token ≈ 4 chars for English, ~3 for mixed Danish/English
def estimate_tokens(text: str) -> int:
    return len(text) // 3


# Model routing — which model to use per mode
DEFAULT_MODEL_MAP = {
    # Heavy reasoning tasks → Opus
    "draft": "claude-opus-4-6",
    "synthesize": "claude-opus-4-6",
    "review": "claude-opus-4-6",
    "transition": "claude-opus-4-6",
    "consistency": "claude-opus-4-6",
    "structure": "claude-opus-4-6",
    # Lighter tasks → Sonnet (much cheaper)
    "chat": "claude-sonnet-4-5-20250929",
    "cite": "claude-sonnet-4-5-20250929",
    "edit_docx": "claude-sonnet-4-5-20250929",
}

# Extended thinking budget per mode (0 = disabled)
DEFAULT_THINKING_MAP = {
    "draft": 10000,
    "synthesize": 12000,
    "review": 8000,
    "transition": 6000,
    "consistency": 6000,
    "structure": 8000,
    "chat": 0,
    "cite": 0,
    "edit_docx": 0,
}

# Max context budget per mode (in estimated tokens)
MAX_CONTEXT_TOKENS = {
    "draft": 80000,
    "synthesize": 100000,
    "review": 60000,
    "transition": 30000,
    "consistency": 40000,
    "structure": 30000,
    "chat": 40000,
    "cite": 30000,
    "edit_docx": 40000,
}


class MBAAgent:
    """Manages interactions with Claude for the MBA paper."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-opus-4-6",
        model_map: dict | None = None,
        thinking_map: dict | None = None,
        max_output_tokens: int = 16000,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.default_model = default_model
        self.model_map = model_map or DEFAULT_MODEL_MAP
        self.thinking_map = thinking_map or DEFAULT_THINKING_MAP
        self.max_output_tokens = max_output_tokens
        self.conversation_history: list[dict] = []

    def get_model(self, mode: str) -> str:
        return self.model_map.get(mode, self.default_model)

    def get_thinking_budget(self, mode: str) -> int:
        return self.thinking_map.get(mode, 0)

    def _build_message(self, user_message: str, context: str, max_context_tokens: int = 80000) -> str:
        """Build the full user message with context, respecting token budget."""
        full = ""
        if context:
            # Trim context if it exceeds budget
            ctx_tokens = estimate_tokens(context)
            if ctx_tokens > max_context_tokens:
                # Truncate from the end (less relevant chunks)
                char_limit = max_context_tokens * 3
                context = context[:char_limit] + "\n\n[... context truncated to fit budget ...]"

            full += f"<retrieved_sources>\n{context}\n</retrieved_sources>\n\n"
        full += user_message
        return full

    def _call(
        self,
        system: str,
        user_message: str,
        context: str = "",
        use_history: bool = False,
        mode: str = "chat",
        model_override: str = "",
        thinking_override: int | None = None,
    ) -> str:
        """
        Non-streaming API call. Returns complete text.
        model_override/thinking_override take precedence over mode defaults.
        """
        model = model_override or self.get_model(mode)
        thinking_budget = thinking_override if thinking_override is not None else self.get_thinking_budget(mode)
        max_ctx = MAX_CONTEXT_TOKENS.get(mode, 80000)

        full_message = self._build_message(user_message, context, max_ctx)

        if use_history:
            messages = self.conversation_history + [
                {"role": "user", "content": full_message}
            ]
        else:
            messages = [{"role": "user", "content": full_message}]

        kwargs = {
            "model": model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": messages,
        }

        # Add extended thinking if budget > 0
        if thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Extended thinking requires higher max_tokens
            kwargs["max_tokens"] = self.max_output_tokens + thinking_budget

        response = self.client.messages.create(**kwargs)

        # Extract text from response (may contain thinking blocks)
        assistant_text = ""
        for block in response.content:
            if block.type == "text":
                assistant_text += block.text

        if use_history:
            self.conversation_history.append(
                {"role": "user", "content": full_message}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_text}
            )
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

        return assistant_text

    def _stream(
        self,
        system: str,
        user_message: str,
        context: str = "",
        use_history: bool = False,
        mode: str = "chat",
        model_override: str = "",
        thinking_override: int | None = None,
    ) -> Generator[dict, None, None]:
        """
        Streaming API call. Yields dicts:
        {"type": "thinking", "text": "..."} for thinking tokens
        {"type": "text", "text": "..."} for output tokens
        {"type": "done", "full_text": "..."} when complete
        {"type": "model", "model": "..."} at start
        model_override/thinking_override take precedence over mode defaults.
        """
        model = model_override or self.get_model(mode)
        thinking_budget = thinking_override if thinking_override is not None else self.get_thinking_budget(mode)
        max_ctx = MAX_CONTEXT_TOKENS.get(mode, 80000)

        full_message = self._build_message(user_message, context, max_ctx)

        if use_history:
            messages = self.conversation_history + [
                {"role": "user", "content": full_message}
            ]
        else:
            messages = [{"role": "user", "content": full_message}]

        kwargs = {
            "model": model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": messages,
        }

        if thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            kwargs["max_tokens"] = self.max_output_tokens + thinking_budget

        yield {"type": "model", "model": model, "thinking_enabled": thinking_budget > 0}

        full_text = ""
        is_thinking = False

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

        if use_history:
            self.conversation_history.append(
                {"role": "user", "content": full_message}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": full_text}
            )
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

        yield {"type": "done", "full_text": full_text}

    # ── Public methods (non-streaming) ──

    def draft(self, instruction: str, context: str) -> str:
        return self._call(prompts.DRAFT_SYSTEM, instruction, context, mode="draft")

    def synthesize(self, topic: str, context: str) -> str:
        return self._call(prompts.SYNTHESIZE_SYSTEM, f"Synthesize the literature on: {topic}", context, mode="synthesize")

    def review(self, draft_text: str, context: str, focus: str = "") -> str:
        msg = f"Review the following draft section:\n\n{draft_text}"
        if focus:
            msg += f"\n\nSpecific focus: {focus}"
        return self._call(prompts.REVIEW_SYSTEM, msg, context, mode="review")

    def cite(self, query: str, context: str) -> str:
        return self._call(prompts.CITE_SYSTEM, f"Find and format citations for: {query}", context, mode="cite")

    def chat(self, message: str, context: str) -> str:
        return self._call(prompts.CHAT_SYSTEM, message, context, use_history=True, mode="chat")

    def edit_docx(self, instruction: str, doc_content: str, context: str) -> str:
        msg = f"INSTRUCTION: {instruction}\n\nCURRENT DOCUMENT CONTENT:\n\n{doc_content}"
        return self._call(prompts.EDIT_DOCX_SYSTEM, msg, context, mode="edit_docx")

    def analyze_transition(self, prev_ending: str, next_beginning: str, red_thread: str, argument_chain: list[str], structure_context: str) -> str:
        msg = f"PREVIOUS SECTION ENDING:\n{prev_ending}\n\nNEXT SECTION BEGINNING:\n{next_beginning}\n\nRED THREAD:\n{red_thread}\n\nARGUMENT CHAIN:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(argument_chain))
        return self._call(prompts.TRANSITION_SYSTEM, msg, structure_context, mode="transition")

    def check_consistency(self, section_text: str, glossary_context: str, structure_context: str, rag_context: str) -> str:
        msg = f"CHECK THIS SECTION FOR CONSISTENCY:\n\n{section_text}\n\n{glossary_context}"
        combined = f"{structure_context}\n\n---\n\n{rag_context}" if rag_context else structure_context
        return self._call(prompts.CONSISTENCY_SYSTEM, msg, combined, mode="consistency")

    def analyze_structure(self, instruction: str, structure_context: str, rag_context: str = "") -> str:
        combined = f"{structure_context}\n\n---\n\n{rag_context}" if rag_context else structure_context
        return self._call(prompts.OUTLINE_SYSTEM, instruction, combined, mode="structure")

    def clear_history(self) -> None:
        self.conversation_history = []

    def draft_with_existing(self, instruction: str, existing_draft: str, context: str) -> str:
        msg = f"INSTRUCTION: {instruction}\n\nEXISTING DRAFT TO WORK FROM:\n\n{existing_draft}"
        return self._call(prompts.DRAFT_SYSTEM, msg, context, mode="draft")

    # ── Streaming public methods ──

    def stream_draft(self, instruction: str, context: str):
        return self._stream(prompts.DRAFT_SYSTEM, instruction, context, mode="draft")

    def stream_synthesize(self, topic: str, context: str):
        return self._stream(prompts.SYNTHESIZE_SYSTEM, f"Synthesize the literature on: {topic}", context, mode="synthesize")

    def stream_review(self, draft_text: str, context: str, focus: str = ""):
        msg = f"Review the following draft section:\n\n{draft_text}"
        if focus: msg += f"\n\nSpecific focus: {focus}"
        return self._stream(prompts.REVIEW_SYSTEM, msg, context, mode="review")

    def stream_chat(self, message: str, context: str):
        return self._stream(prompts.CHAT_SYSTEM, message, context, use_history=True, mode="chat")

    def stream_cite(self, query: str, context: str):
        return self._stream(prompts.CITE_SYSTEM, f"Find and format citations for: {query}", context, mode="cite")

    def stream_generic(self, system: str, message: str, context: str, mode: str = "chat", use_history: bool = False):
        """Generic streaming method for any mode."""
        return self._stream(system, message, context, use_history=use_history, mode=mode)
