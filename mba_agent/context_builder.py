"""
Context builder: assembles context for LLM calls.
Two strategies: RAG (ChromaDB retrieval for Claude) and full-text (for Gemini).
"""

import re
from pathlib import Path


class ContextBuilder:
    """Builds context for LLM calls using either RAG or full-text strategy."""

    def __init__(self, store, citations, cfg: dict):
        self.store = store         # PaperStore (ChromaDB) — may be None for Gemini-only
        self.citations = citations
        self.cfg = cfg
        self._full_text_cache = None
        self._all_source_files = []

    def build_rag_context(
        self,
        query: str,
        mode: str,
        section_id: str,
        ps,
        key_sources: list | None = None,
    ) -> tuple[str, list[str]]:
        """
        RAG-based context building for Claude.
        If key_sources provided, boosts those sources in retrieval.
        """
        search_query = query[:500]
        n_results = self.cfg.get("max_retrieval_chunks", 25)

        if key_sources:
            # Source-biased retrieval: priority pass on key sources, then open
            priority_results = []
            for src in key_sources[:5]:
                results = self.store.search(search_query, n_results=5, source_filter=src)
                priority_results.extend(results)

            open_results = self.store.search(search_query, n_results=n_results)

            # Merge: priority first, then open (deduplicated)
            seen = set()
            combined = []
            for r in priority_results + open_results:
                key = (r["source_file"], r.get("page_start", 0))
                if key not in seen:
                    seen.add(key)
                    combined.append(r)

            # Build context string from combined results
            context_parts = []
            for r in combined[:n_results]:
                entry = f"{r.get('source_tag', '')}\n{r['text']}"
                context_parts.append(entry)

            rag_context = "\n\n---\n\n".join(context_parts)
        else:
            rag_context = self.store.build_context(search_query, n_results=n_results)

        source_files = list(set(re.findall(r'\[SOURCE: (.+?),', rag_context)))
        return rag_context, source_files

    def build_full_context(
        self, mode: str, section_id: str, ps,
    ) -> tuple[str, list[str]]:
        """
        Full-text context for Gemini. Loads all source texts without retrieval.
        Fits within Gemini's 1M token context window.
        """
        if self._full_text_cache is None:
            self._load_all_texts()

        return self._full_text_cache, list(self._all_source_files)

    def _load_all_texts(self) -> None:
        """Extract and clean all documents, concatenate into one string."""
        from .ingest import extract_full_text, extract_pdf_text

        papers_dir = self.cfg.get("papers_dir", "./papers")
        papers_path = Path(papers_dir)
        parts = []
        self._all_source_files = []

        # PDFs
        for pdf in sorted(papers_path.glob("**/*.pdf")):
            try:
                pages = extract_pdf_text(str(pdf), strip_references=True)
                if pages:
                    text = "\n\n".join(
                        f"[SOURCE: {pdf.name}, page {p}]\n{t}" for p, t in pages
                    )
                    parts.append(text)
                    self._all_source_files.append(pdf.name)
            except Exception:
                pass

        # DOCX files
        for docx in sorted(papers_path.glob("**/*.docx")):
            try:
                from docx import Document
                doc = Document(str(docx))
                text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
                if text:
                    parts.append(f"[SOURCE: {docx.name}]\n{text}")
                    self._all_source_files.append(docx.name)
            except Exception:
                pass

        # Text/Markdown files
        for ext in ("*.txt", "*.md"):
            for txt in sorted(papers_path.glob(f"**/{ext}")):
                try:
                    content = txt.read_text(encoding="utf-8", errors="replace")
                    if content.strip():
                        parts.append(f"[SOURCE: {txt.name}]\n{content}")
                        self._all_source_files.append(txt.name)
                except Exception:
                    pass

        self._full_text_cache = "\n\n===\n\n".join(parts)

        # Safety check: if too large, warn (Gemini 1M ≈ ~4M chars)
        if len(self._full_text_cache) > 3_500_000:
            # Trim to most recent/important files (keep first 3.5M chars)
            self._full_text_cache = self._full_text_cache[:3_500_000] + \
                "\n\n[... remaining sources truncated to fit context window ...]"

    def invalidate_cache(self) -> None:
        """Call after new files are ingested."""
        self._full_text_cache = None
        self._all_source_files = []
