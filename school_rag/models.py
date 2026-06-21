from __future__ import annotations

from typing import NotRequired, TypedDict


class Chunk(TypedDict):
    chunk_id: str
    text: str
    source: str
    document_id: str
    title: str
    section: str
    page: int
    record_id: str


class RetrievedChunk(Chunk):
    score: float
    score_type: str


class Source(TypedDict):
    number: int
    chunk_id: str
    source: str
    title: str
    section: str
    page: int


class RAGState(TypedDict):
    original_query: str
    rewritten_query: NotRequired[str]
    retrieved_chunks: NotRequired[list[RetrievedChunk]]
    answer: NotRequired[str]
    sources: NotRequired[list[Source]]
    error: NotRequired[str]
