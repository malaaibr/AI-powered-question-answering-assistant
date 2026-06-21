from __future__ import annotations

from typing import Protocol

from .models import Chunk, RetrievedChunk
from .services import Embedder


class VectorStore(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


class ChromaVectorStore:
    """Persistent cosine-distance Chroma store with explicit similarity conversion."""

    def __init__(self, path: str, collection_name: str, embedder: Embedder) -> None:
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def index(self, chunks: list[Chunk], fingerprint: str, batch_size: int = 64) -> bool:
        metadata = self.collection.metadata or {}
        if metadata.get("fingerprint") == fingerprint and self.collection.count() == len(chunks):
            return False
        if self.collection.count():
            self.collection.delete(ids=self.collection.get(include=[])["ids"])
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [item["text"] for item in batch]
            metadatas = [{key: value for key, value in item.items() if key != "text"} for item in batch]
            self.collection.add(
                ids=[item["chunk_id"] for item in batch],
                documents=texts,
                metadatas=metadatas,
                embeddings=self.embedder.embed_documents(texts),
            )
        self.collection.modify(metadata={"hnsw:space": "cosine", "fingerprint": fingerprint})
        return True

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        found: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            found.append(
                {
                    **metadata,
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": max(-1.0, min(1.0, 1.0 - float(distance))),
                    "score_type": "cosine_similarity",
                }
            )
        return sorted(found, key=lambda item: item["score"], reverse=True)
