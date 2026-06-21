from __future__ import annotations

import os
from typing import Protocol, Sequence

from .prompts import GROUNDING_PROMPT, REWRITE_PROMPT


class LLM(Protocol):
    def rewrite(self, query: str) -> str: ...
    def generate(self, prompt: str) -> str: ...


class Embedder(Protocol):
    name: str
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OpenAILLM:
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required to rewrite and answer questions; copy .env.example to .env")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def rewrite(self, query: str) -> str:
        return self._call(REWRITE_PROMPT, query)

    def generate(self, prompt: str) -> str:
        return self._call(GROUNDING_PROMPT, prompt)

    def _call(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
