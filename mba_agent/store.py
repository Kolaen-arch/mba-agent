"""
Vector store: ChromaDB wrapper with multilingual embeddings + BM25 hybrid search.
Handles storage, retrieval, and similarity search across the paper library.
"""

import json
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


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


class PaperStore:
    """Manages the vector database of paper chunks."""

    def __init__(
        self,
        persist_dir: str = "./.chroma_db",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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

    @property
    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        """Embed and store chunks in batches."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            ids = [f"{c.source_file}_{c.chunk_index}" for c in batch]

            # Generate embeddings
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

            # Metadata for each chunk
            metadatas = [
                {
                    "source_file": c.source_file,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "chunk_index": c.chunk_index,
                    "source_tag": c.source_tag,
                    "label": c.label or "",
                }
                for c in batch
            ]

            # Upsert to handle re-ingestion
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
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
        Uses hybrid search (semantic + BM25) when available.
        Deduplicates and sorts by relevance.
        """
        if use_hybrid and HAS_BM25:
            results = self.search_hybrid(query, n_results=n_results)
        else:
            results = self.search(query, n_results=n_results)

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

    def search_hybrid(
        self,
        query: str,
        n_results: int = 25,
        alpha: float = 0.7,
    ) -> list[dict]:
        """
        Hybrid search: alpha * semantic_score + (1-alpha) * bm25_score.
        Falls back to pure semantic search if BM25 is unavailable.
        """
        # Semantic search
        semantic_results = self.search(query, n_results=min(n_results * 2, 50))

        if not HAS_BM25 or alpha >= 1.0:
            return semantic_results[:n_results]

        # Build BM25 index if needed
        if self._bm25_index is None:
            self._build_bm25_index()

        if not self._bm25_index or not self._bm25_corpus:
            return semantic_results[:n_results]

        # BM25 search
        query_tokens = _tokenize(query)
        bm25_scores = self._bm25_index.get_scores(query_tokens)

        # Normalize BM25 scores to 0-1
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_map = {}
        for i, entry in enumerate(self._bm25_corpus):
            bm25_map[entry["id"]] = bm25_scores[i] / max_bm25

        # Normalize semantic scores (distance → similarity, 0-1)
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        max_dist = max(r["distance"] for r in semantic_results) if semantic_results else 1.0
        if max_dist == 0:
            max_dist = 1.0

        # Combine scores
        combined = {}
        for r in semantic_results:
            doc_id = f"{r['source_file']}_{r.get('chunk_index', 0)}"
            semantic_score = 1.0 - (r["distance"] / max_dist)
            bm25_score = bm25_map.get(doc_id, 0.0)
            combined[doc_id] = {
                **r,
                "score": alpha * semantic_score + (1 - alpha) * bm25_score,
            }

        # Also check top BM25 results that might not be in semantic results
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_results]
        for i in top_bm25_indices:
            entry = self._bm25_corpus[i]
            doc_id = entry["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "text": entry["text"],
                    "source_file": entry["metadata"]["source_file"],
                    "page_start": entry["metadata"].get("page_start", 0),
                    "page_end": entry["metadata"].get("page_end", 0),
                    "source_tag": entry["metadata"].get("source_tag", ""),
                    "distance": 1.0,  # Unknown semantic distance
                    "score": (1 - alpha) * (bm25_scores[i] / max_bm25),
                }

        # Sort by combined score, return top n
        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:n_results]

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
