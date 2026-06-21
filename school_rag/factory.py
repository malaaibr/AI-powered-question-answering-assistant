from __future__ import annotations

from .config import Settings
from .graph import RAGWorkflow
from .services import OpenAILLM, SentenceTransformerEmbedder
from .store import ChromaVectorStore


def build_components(settings: Settings):
    settings.validate()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = ChromaVectorStore(str(settings.vector_store_path), settings.collection_name, embedder)
    return embedder, store


def build_workflow(settings: Settings) -> RAGWorkflow:
    _, store = build_components(settings)
    llm = OpenAILLM(settings.llm_model, settings.temperature)
    return RAGWorkflow(llm, store, settings.top_k)
