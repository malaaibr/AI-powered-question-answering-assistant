from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_path: Path = Path("data/private_school_content.json")
    links_path: Path = Path("data/private_school_links.txt")
    vector_store_path: Path = Path(".chroma")
    collection_name: str = "egyptian_private_universities"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 1200
    chunk_overlap: int = 180
    top_k: int = 5
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            data_path=Path(os.getenv("DATA_PATH", str(cls.data_path))),
            links_path=Path(os.getenv("LINKS_PATH", str(cls.links_path))),
            vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", str(cls.vector_store_path))),
            collection_name=os.getenv("CHROMA_COLLECTION", cls.collection_name),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            chunk_size=_positive_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_nonnegative_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_positive_int("TOP_K", cls.top_k),
            temperature=float(os.getenv("TEMPERATURE", str(cls.temperature))),
        )

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value
