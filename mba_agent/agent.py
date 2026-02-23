"""
Agent: LLM interaction orchestrator.
Delegates to backend adapters (Claude, Gemini).
Handles model routing, context assembly, conversation history.
"""

from typing import Generator

from .llm_adapter import LLMBackend
from . import prompts


# Token estimation: ~4 chars/token for English, ~3.5 for Danish (more compound words, æøå)
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Detect Danish by checking for æøå frequency
    sample = text[:2000].lower()
    danish_chars = sum(1 for c in sample if c in 'æøåéü')
    if danish_chars > 3:
        return len(text) * 10 // 35  # ~3.5 chars/token for Danish
    return len(text) // 4  # ~4 chars/token for English


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
    """Manages LLM interactions for the MBA paper. Backend-agnostic."""

    def __init__(
        self,
        backends: dict[str, LLMBackend],
        default_model: str = "claude-opus-4-6",
        model_map: dict | None = None,
        thinking_map: dict | None = None,
        max_output_tokens: int = 16000,
    ):
        self.backends = backends
        self.default_model = default_model
        self.model_map = model_map or DEFAULT_MODEL_MAP
        self.thinking_map = thinking_map or DEFAULT_THINKING_MAP
        self.max_output_tokens = max_output_tokens
        self.conversation_history: list[dict] = []

    def _get_backend(self, model: str) -> LLMBackend:
        """Find the backend that handles this model."""
        if model in self.backends:
            return self.backends[model]
        # Fallback: match by provider prefix
        for key, backend in self.backends.items():
            if model.startswith(key.split("-")[0]):
                return backend
        # Last resort: default model's backend
        return self.backends.get(self.default_model, next(iter(self.backends.values())))

    def get_model(self, mode: str) -> str:
        return self.model_map.get(mode, self.default_model)

    def get_thinking_budget(self, mode: str) -> int:
        return self.thinking_map.get(mode, 0)

    def available_models(self) -> list[str]:
        """Return list of available model names."""
        return list(self.backends.keys())

    def _build_message(self, user_message: str, context: str, max_context_tokens: int = 80000) -> str:
        """
        Build the full user message with context, respecting token budget.
        Uses structured truncation: structure/citation context is never trimmed,
        only RAG chunks are removed from the bottom (lowest relevance first).
        """
        full = ""
        if context:
            ctx_tokens = estimate_tokens(context)
            if ctx_tokens > max_context_tokens:
                context = self._smart_truncate(context, max_context_tokens)

            full += f"<retrieved_sources>\n{context}\n</retrieved_sources>\n\n"
        full += user_message
        return full

    @staticmethod
    def _smart_truncate(context: str, max_tokens: int) -> str:
        """
        Priority-based truncation. Structure/citation context preserved,
        RAG chunks trimmed from the bottom (lowest relevance).
        """
        parts = context.split("\n\n---\n\n", 1)

        if len(parts) == 2:
            priority_ctx = parts[0]
            rag_ctx = parts[1]
        else:
            priority_ctx = ""
            rag_ctx = context

        priority_tokens = estimate_tokens(priority_ctx) if priority_ctx else 0
        remaining = max_tokens - priority_tokens

        if remaining <= 0:
            char_limit = max_tokens * 3
            return priority_ctx[:char_limit] + "\n\n[... context truncated ...]"

        rag_chunks = rag_ctx.split("\n\n")
        kept = []
        used = 0
        for chunk in rag_chunks:
            chunk_tokens = estimate_tokens(chunk)
            if used + chunk_tokens > remaining:
                break
            kept.append(chunk)
            used += chunk_tokens

        trimmed_rag = "\n\n".join(kept)
        if len(kept) < len(rag_chunks):
            trimmed_rag += f"\n\n[... {len(rag_chunks) - len(kept)} lower-relevance chunks trimmed ...]"

        if priority_ctx:
            return f"{priority_ctx}\n\n---\n\n{trimmed_rag}"
        return trimmed_rag

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
        Delegates to the appropriate backend based on model selection.
        """
        model = model_override or self.get_model(mode)
        thinking_budget = thinking_override if thinking_override is not None else self.get_thinking_budget(mode)
        max_ctx = MAX_CONTEXT_TOKENS.get(mode, 80000)

        full_message = self._build_message(user_message, context, max_ctx)

        history = None
        if use_history:
            history = list(self.conversation_history)

        backend = self._get_backend(model)
        result = backend.call(
            system=system,
            user_message=full_message,
            mode=mode,
            thinking_budget=thinking_budget,
            max_output_tokens=self.max_output_tokens,
            history=history,
        )

        if use_history:
            self.conversation_history.append(
                {"role": "user", "content": full_message}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": result}
            )
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

        return result

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
        Streaming API call. Delegates to the appropriate backend.
        Yields dicts with standard event types.
        """
        model = model_override or self.get_model(mode)
        thinking_budget = thinking_override if thinking_override is not None else self.get_thinking_budget(mode)
        max_ctx = MAX_CONTEXT_TOKENS.get(mode, 80000)

        full_message = self._build_message(user_message, context, max_ctx)

        history = None
        if use_history:
            history = list(self.conversation_history)

        backend = self._get_backend(model)
        full_text = ""

        for chunk in backend.stream(
            system=system,
            user_message=full_message,
            mode=mode,
            thinking_budget=thinking_budget,
            max_output_tokens=self.max_output_tokens,
            history=history,
        ):
            if chunk["type"] == "done":
                full_text = chunk.get("full_text", "")
            yield chunk

        if use_history:
            self.conversation_history.append(
                {"role": "user", "content": full_message}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": full_text}
            )
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

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
