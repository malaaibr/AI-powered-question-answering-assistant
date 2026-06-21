from __future__ import annotations

from school_rag.models import RetrievedChunk


def retrieved_chunk(text: str = "Nile Technical University was founded in 2018.", score: float = 0.9) -> RetrievedChunk:
    return {
        "chunk_id": "chunk-1",
        "text": text,
        "source": "https://example.edu/nile/about",
        "document_id": "record-0001",
        "title": "Nile Technical University",
        "section": "About",
        "page": 1,
        "record_id": "record-0001",
        "score": score,
        "score_type": "cosine_similarity",
    }
