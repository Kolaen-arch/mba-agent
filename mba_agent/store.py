"""
Vector store: ChromaDB wrapper with multilingual embeddings.
Handles storage, retrieval, and similarity search across the paper library.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .ingest import Chunk


class PaperStore:
    """Manages the vector database of paper chunks."""

    def __init__(
        self,
        persist_dir: str = "./.chroma_db",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.persist_dir = persist_dir
        self.embedder = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name="mba_papers",
            metadata={"hnsw:space": "cosine"},
        )

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
    ) -> str:
        """
        Search and build a formatted context string for the LLM.
        Deduplicates and sorts by relevance.
        """
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

    def list_sources(self) -> list[str]:
        """List all unique source files in the store."""
        # Get all metadata
        result = self.collection.get(include=["metadatas"])
        sources = set()
        if result["metadatas"]:
            for meta in result["metadatas"]:
                sources.add(meta["source_file"])
        return sorted(sources)

    def clear(self) -> None:
        """Delete all data and recreate collection."""
        self.client.delete_collection("mba_papers")
        self.collection = self.client.get_or_create_collection(
            name="mba_papers",
            metadata={"hnsw:space": "cosine"},
        )
