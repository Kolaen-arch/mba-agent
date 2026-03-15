"""
Context builder: assembles context for LLM calls.
Two strategies: RAG (ChromaDB retrieval for Claude) and full-text (for Gemini).
Includes HyDE, doc-type biasing, and multi-query expansion.
"""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Document-type retrieval bias: weight multiplier on RRF score per mode
DOC_TYPE_WEIGHTS = {
    "draft":      {"own_work": 1.5, "textbook": 1.3, "article": 1.0, "case": 1.0},
    "synthesize": {"article": 1.3, "textbook": 1.2, "own_work": 0.8, "case": 1.0},
    "review":     {"own_work": 1.5, "article": 1.0, "textbook": 0.8, "case": 1.0},
    "chat":       {"article": 1.0, "textbook": 1.0, "own_work": 1.0, "case": 1.0},
    "cite":       {"article": 1.3, "textbook": 1.0, "own_work": 0.7, "case": 1.0},
}


class ContextBuilder:
    """Builds context for LLM calls using either RAG or full-text strategy."""

    def __init__(self, store, citations, cfg: dict, hyde_backend=None):
        self.store = store         # PaperStore (ChromaDB) — may be None for Gemini-only
        self.citations = citations
        self.cfg = cfg
        self._full_text_cache = None
        self._all_source_files = []

        # Retrieval upgrades (config-gated)
        retrieval_cfg = cfg.get("retrieval", {})
        self._hyde_enabled = retrieval_cfg.get("hyde_enabled", False)
        self._hyde_modes = retrieval_cfg.get("hyde_modes", ["draft", "synthesize"])
        self._doc_type_bias = retrieval_cfg.get("doc_type_bias", False)
        self._query_expansion = retrieval_cfg.get("query_expansion", False)

        # LLM backend for HyDE / query expansion (cheapest available model)
        self._hyde_backend = hyde_backend

        # Knowledge graph (set externally when GraphRAG is enabled)
        self.knowledge_graph = None

    def build_rag_context(
        self,
        query: str,
        mode: str,
        section_id: str,
        ps,
        key_sources: list | None = None,
    ) -> tuple[str, list[str]]:
        """
        RAG-based context building.
        Pipeline: [HyDE] → [graph boost] → hybrid search → [doc-type bias] → format.
        """
        search_query = query[:500]
        n_results = self.cfg.get("max_retrieval_chunks", 25)

        # ── HyDE: generate hypothetical paragraph for embedding ──
        if (self._hyde_enabled and self._hyde_backend
                and mode in self._hyde_modes):
            hyde_text = self._generate_hyde(search_query)
            if hyde_text:
                search_query = hyde_text

        # ── GraphRAG: boost sources connected to query entities ──
        if self.knowledge_graph and hasattr(self.knowledge_graph, 'find_connected'):
            graph_entities = self._extract_query_entities(query)
            if graph_entities:
                graph_sources = self.knowledge_graph.find_connected(graph_entities)
                key_sources = list(set((key_sources or []) + graph_sources))

        # ── Query expansion: multi-query retrieval ──
        if (self._query_expansion and self._hyde_backend
                and mode in ("draft", "synthesize", "review")):
            results = self._expanded_retrieval(query, search_query, n_results)
        elif key_sources:
            results = self._source_biased_retrieval(search_query, key_sources, n_results)
        else:
            results = self._standard_retrieval(search_query, n_results)

        # ── Doc-type bias: adjust scores per mode ──
        if self._doc_type_bias and results:
            results = self._apply_doc_type_bias(results, mode)

        # ── Format context string ──
        rag_context = self._format_results(results, n_results)

        source_files = list(set(re.findall(r'\[SOURCE: (.+?),', rag_context)))
        return rag_context, source_files

    def _standard_retrieval(self, query: str, n_results: int) -> list[dict]:
        """Standard hybrid retrieval with optional reranking."""
        if hasattr(self.store, 'reranker') and self.store.reranker:
            # Retrieve more candidates for reranking
            results = self.store.search_hybrid(query, n_results=max(n_results * 2, 50))
            results = self.store.reranker.rerank(query, results, top_k=n_results)
        else:
            results = self.store.search_hybrid(query, n_results=n_results)
        return results

    def _source_biased_retrieval(
        self, query: str, key_sources: list, n_results: int
    ) -> list[dict]:
        """Priority retrieval from key sources, then open retrieval."""
        priority_results = []
        for src in key_sources[:5]:
            results = self.store.search(query, n_results=5, source_filter=src)
            priority_results.extend(results)

        open_results = self.store.search_hybrid(query, n_results=n_results)

        # Merge: priority first, then open (deduplicated)
        seen = set()
        combined = []
        for r in priority_results + open_results:
            key = (r["source_file"], r.get("page_start", 0))
            if key not in seen:
                seen.add(key)
                combined.append(r)

        # Rerank the combined set if available
        if hasattr(self.store, 'reranker') and self.store.reranker:
            combined = self.store.reranker.rerank(query, combined, top_k=n_results)

        return combined[:n_results]

    def _expanded_retrieval(
        self, original_query: str, search_query: str, n_results: int
    ) -> list[dict]:
        """Multi-query retrieval: expand query into alternatives, merge via RRF."""
        queries = self._expand_query(original_query)
        if len(queries) <= 1:
            return self._standard_retrieval(search_query, n_results)

        # Retrieve for each query variant
        all_ranked: list[list[dict]] = []
        for q in queries:
            results = self.store.search_hybrid(q, n_results=30)
            all_ranked.append(results)

        # RRF merge across query variants
        merged = self._rrf_merge_multi(all_ranked)

        # Rerank the merged set
        if hasattr(self.store, 'reranker') and self.store.reranker:
            merged = self.store.reranker.rerank(original_query, merged, top_k=n_results)

        return merged[:n_results]

    def _generate_hyde(self, query: str) -> str | None:
        """Generate hypothetical ideal paragraph for HyDE embedding."""
        try:
            response = self._hyde_backend.call(
                system="You are a concise academic writer. Write dense, specific text.",
                user_message=(
                    "Write a single dense academic paragraph (150-200 words) that would "
                    "perfectly answer this query. Include specific author names, theories, "
                    "and concepts that a real academic paper would use. Do not fabricate "
                    "citations — use plausible academic language and terminology.\n\n"
                    f"Query: {query}"
                ),
                mode="chat",
                thinking_budget=0,
                max_output_tokens=400,
            )
            if response and len(response) > 50:
                log.info("HyDE generated %d chars for query", len(response))
                return response[:800]
        except Exception as e:
            log.warning("HyDE generation failed: %s", e)
        return None

    def _expand_query(self, query: str) -> list[str]:
        """Generate 2-3 alternative search queries for multi-query retrieval."""
        try:
            response = self._hyde_backend.call(
                system="You are a research librarian. Return one query per line, nothing else.",
                user_message=(
                    "Generate 3 alternative academic search queries for this topic. "
                    "Use different terminology, synonyms, and related concepts.\n\n"
                    f"Original: {query}"
                ),
                mode="chat",
                thinking_budget=0,
                max_output_tokens=200,
            )
            alternatives = [
                q.strip().lstrip("0123456789.-) ")
                for q in response.strip().split("\n")
                if q.strip() and len(q.strip()) > 10
            ]
            return [query] + alternatives[:2]
        except Exception as e:
            log.warning("Query expansion failed: %s", e)
            return [query]

    def _extract_query_entities(self, query: str) -> list[str]:
        """Simple NER: extract likely academic entities from query text."""
        # Match capitalized multi-word phrases (author names, theories)
        entities = re.findall(r'(?:[A-Z][a-z]+(?:\s+(?:&|and)\s+)?)+[A-Z][a-z]+', query)
        # Also match known patterns like "X's Y" (e.g. "Pine & Gilmore's experience economy")
        possessives = re.findall(r'([A-Z][a-z]+(?:\s+(?:&|and)\s+[A-Z][a-z]+)?)\s*\'s', query)
        return list(set(entities + possessives))

    @staticmethod
    def _apply_doc_type_bias(results: list[dict], mode: str) -> list[dict]:
        """Apply document-type weight multiplier to retrieval scores."""
        weights = DOC_TYPE_WEIGHTS.get(mode, DOC_TYPE_WEIGHTS["chat"])
        for r in results:
            doc_type = r.get("doc_type", "article")
            multiplier = weights.get(doc_type, 1.0)
            if "score" in r:
                r["score"] *= multiplier
            if "rerank_score" in r:
                r["rerank_score"] *= multiplier
        # Re-sort by whichever score is present
        sort_key = "rerank_score" if "rerank_score" in results[0] else "score"
        if sort_key in results[0]:
            results.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
        return results

    @staticmethod
    def _rrf_merge_multi(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """RRF merge across multiple ranked lists (from query expansion)."""
        scores: dict[str, float] = {}
        data: dict[str, dict] = {}

        for ranked in ranked_lists:
            for rank, r in enumerate(ranked):
                doc_id = f"{r['source_file']}_{r.get('chunk_index', 0)}"
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
                if doc_id not in data:
                    data[doc_id] = r

        fused = []
        for doc_id, score in scores.items():
            entry = {**data[doc_id], "score": score}
            fused.append(entry)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused

    @staticmethod
    def _format_results(results: list[dict], max_results: int) -> str:
        """Format retrieval results into context string."""
        context_parts = []
        seen_texts = set()

        for r in results[:max_results]:
            text_hash = hash(r["text"][:200])
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)
            entry = f"{r.get('source_tag', '')}\n{r['text']}"
            context_parts.append(entry)

        return "\n\n---\n\n".join(context_parts)

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
                pages = extract_pdf_text(str(pdf), strip_references=True, detect_headings=False)
                if pages:
                    text = "\n\n".join(
                        f"[SOURCE: {pdf.name}, page {p}]\n{t}" for p, t, *_ in pages
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
            self._full_text_cache = self._full_text_cache[:3_500_000] + \
                "\n\n[... remaining sources truncated to fit context window ...]"

    def invalidate_cache(self) -> None:
        """Call after new files are ingested."""
        self._full_text_cache = None
        self._all_source_files = []
