"""
Vector store: ChromaDB wrapper with multilingual embeddings + BM25 hybrid search.
Handles storage, retrieval, and similarity search across the paper library.
"""

import json
import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from .ingest import Chunk

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


class Reranker:
    """Cross-encoder reranker for fine-grained query-chunk relevance scoring.
    Lazy-loaded: model only downloaded/loaded on first use."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        log.info("Loading reranker model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name, device="cpu")

    def rerank(self, query: str, results: list[dict], top_k: int = 15) -> list[dict]:
        """Score (query, chunk) pairs and return top_k by cross-encoder score."""
        if not results:
            return results
        self._load()

        pairs = [(query, r["text"]) for r in results]
        scores = self._model.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


class PaperStore:
    """Manages the vector database of paper chunks."""

    def __init__(
        self,
        persist_dir: str = "./.chroma_db",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        reranker_enabled: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.persist_dir = persist_dir
        # Workaround: ROCm PyTorch on Windows lacks torch.distributed —
        # SentenceTransformer's auto device detection crashes. Force device.
        try:
            import torch
            if not hasattr(torch.distributed, 'is_initialized'):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.embedder = SentenceTransformer(embedding_model, device=device)
            else:
                self.embedder = SentenceTransformer(embedding_model)
        except Exception:
            self.embedder = SentenceTransformer(embedding_model, device="cpu")

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name="mba_papers",
            metadata={"hnsw:space": "cosine"},
        )

        self._sources_cache: list[str] | None = None
        self._bm25_index: "BM25Okapi | None" = None
        self._bm25_corpus: list[dict] = []  # [{id, text, metadata}]

        # Cross-encoder reranker (lazy-loaded on first query)
        self.reranker: Reranker | None = Reranker(reranker_model) if reranker_enabled else None

    @property
    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        """Embed and store chunks in batches.
        Embeds contextual headers (document + section) for better retrieval.
        Deduplicates by ID within each batch (same filename in multiple dirs)."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Deduplicate within batch by ID (same file in different subdirs)
            seen_ids = set()
            deduped = []
            for c in batch:
                cid = f"{c.source_file}_{c.chunk_index}"
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    deduped.append(c)
            batch = deduped

            # Embed the contextual header version for better retrieval
            embed_texts = [c.embedding_text for c in batch]
            # Store the raw text as the document (for display/context building)
            store_texts = [c.text for c in batch]
            ids = [f"{c.source_file}_{c.chunk_index}" for c in batch]

            # Generate embeddings from contextual text
            embeddings = self.embedder.encode(embed_texts, show_progress_bar=False).tolist()

            # Metadata for each chunk
            metadatas = [
                {
                    "source_file": c.source_file,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "chunk_index": c.chunk_index,
                    "source_tag": c.source_tag,
                    "label": c.label or "",
                    "section_header": c.section_header or "",
                    "doc_type": c.doc_type or "",
                }
                for c in batch
            ]

            # Upsert to handle re-ingestion
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=store_texts,
                metadatas=metadatas,
            )
            print(f"  Stored chunks {i+1}-{min(i+batch_size, len(chunks))} / {len(chunks)}")
        self._sources_cache = None  # Invalidate cache after adding chunks
        self._bm25_index = None  # Invalidate BM25 index

    def search(
        self,
        query: str,
        n_results: int = 25,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Search for relevant chunks.
        Returns list of dicts with text, metadata, and distance.
        """
        query_embedding = self.embedder.encode([query]).tolist()

        where_filter = None
        if source_filter:
            where_filter = {"source_file": source_filter}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "text": doc,
                    "source_file": meta["source_file"],
                    "page_start": meta["page_start"],
                    "page_end": meta["page_end"],
                    "source_tag": meta["source_tag"],
                    "chunk_index": meta.get("chunk_index", 0),
                    "doc_type": meta.get("doc_type", ""),
                    "distance": dist,
                })

        return output

    def build_context(
        self,
        query: str,
        n_results: int = 25,
        max_chars: int = 480000,  # ~120K tokens
        use_hybrid: bool = True,
    ) -> str:
        """
        Search and build a formatted context string for the LLM.
        Pipeline: hybrid search (retrieve 50) → optional rerank (top 15) → format.
        Deduplicates and sorts by relevance.
        """
        # With reranker: retrieve more, rerank to fewer high-quality results
        if self.reranker:
            retrieve_n = max(n_results * 2, 50)
        else:
            retrieve_n = n_results

        if use_hybrid and HAS_BM25:
            results = self.search_hybrid(query, n_results=retrieve_n)
        else:
            results = self.search(query, n_results=retrieve_n)

        # Cross-encoder reranking for fine-grained relevance
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=n_results)

        context_parts = []
        total_chars = 0
        seen_texts = set()

        for r in results:
            # Deduplicate
            text_hash = hash(r["text"][:200])
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)

            entry = f"{r['source_tag']}\n{r['text']}"
            if total_chars + len(entry) > max_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(context_parts)

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in ChromaDB."""
        if not HAS_BM25:
            return
        result = self.collection.get(include=["documents", "metadatas"])
        if not result["documents"]:
            self._bm25_corpus = []
            self._bm25_index = None
            return

        self._bm25_corpus = []
        tokenized = []
        for doc_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            self._bm25_corpus.append({"id": doc_id, "text": text, "metadata": meta})
            tokenized.append(_tokenize(text))

        self._bm25_index = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, n_results: int = 50) -> list[dict]:
        """BM25 keyword search. Returns ranked results with metadata."""
        if not HAS_BM25:
            return []

        if self._bm25_index is None:
            self._build_bm25_index()

        if not self._bm25_index or not self._bm25_corpus:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25_index.get_scores(query_tokens)

        # Rank by score descending
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_results]

        results = []
        for rank, i in enumerate(ranked_indices):
            if scores[i] <= 0:
                break
            entry = self._bm25_corpus[i]
            meta = entry["metadata"]
            results.append({
                "text": entry["text"],
                "source_file": meta["source_file"],
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "source_tag": meta.get("source_tag", ""),
                "doc_type": meta.get("doc_type", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "bm25_score": float(scores[i]),
                "bm25_rank": rank,
                "_doc_id": entry["id"],
            })

        return results

    def search_hybrid(
        self,
        query: str,
        n_results: int = 25,
        retrieve_depth: int = 50,
    ) -> list[dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).
        Retrieves `retrieve_depth` from each system, fuses by rank position.
        RRF score = sum(1 / (k + rank_i)) — normalization-free.
        Falls back to pure semantic search if BM25 is unavailable.
        """
        k = 60  # RRF constant (Cormack et al. 2009)

        # Retrieve from both systems
        semantic_results = self.search(query, n_results=retrieve_depth)

        if not HAS_BM25:
            return semantic_results[:n_results]

        bm25_results = self._bm25_search(query, n_results=retrieve_depth)
        if not bm25_results:
            return semantic_results[:n_results]

        # Build RRF scores — keyed by doc_id
        rrf_scores: dict[str, float] = {}
        result_data: dict[str, dict] = {}

        # Score semantic results by rank position
        for rank, r in enumerate(semantic_results):
            doc_id = f"{r['source_file']}_{r.get('chunk_index', 0)}"
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            if doc_id not in result_data:
                result_data[doc_id] = r

        # Score BM25 results by rank position
        for rank, r in enumerate(bm25_results):
            doc_id = r["_doc_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            if doc_id not in result_data:
                result_data[doc_id] = r

        # Merge: attach RRF score to each result
        fused = []
        for doc_id, score in rrf_scores.items():
            entry = {**result_data[doc_id], "score": score}
            entry.pop("_doc_id", None)
            entry.pop("bm25_score", None)
            entry.pop("bm25_rank", None)
            fused.append(entry)

        # Sort by RRF score descending
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:n_results]

    def list_sources(self) -> list[str]:
        """List all unique source files in the store. Cached after first call."""
        if self._sources_cache is not None:
            return self._sources_cache
        result = self.collection.get(include=["metadatas"])
        sources = set()
        if result["metadatas"]:
            for meta in result["metadatas"]:
                sources.add(meta["source_file"])
        self._sources_cache = sorted(sources)
        return self._sources_cache

    def list_sources_with_labels(self) -> list[dict]:
        """List all unique source files with their labels and chunk counts."""
        result = self.collection.get(include=["metadatas"])
        source_info: dict[str, dict] = {}
        if result["metadatas"]:
            for meta in result["metadatas"]:
                fname = meta["source_file"]
                if fname not in source_info:
                    source_info[fname] = {
                        "file": fname,
                        "label": meta.get("label", ""),
                        "chunks": 0,
                    }
                source_info[fname]["chunks"] += 1
        return sorted(source_info.values(), key=lambda x: x["file"])

    def clear(self) -> None:
        """Delete all data and recreate collection."""
        self.client.delete_collection("mba_papers")
        self.collection = self.client.get_or_create_collection(
            name="mba_papers",
            metadata={"hnsw:space": "cosine"},
        )
        self._sources_cache = None
        self._bm25_index = None
        self._bm25_corpus = []
