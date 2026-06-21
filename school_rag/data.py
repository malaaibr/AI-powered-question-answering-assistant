from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

from .models import Chunk


BOILERPLATE = {
    "this institution has verified its data",
    "this institution has not verified its data",
    "verify it now!",
    "apply for admissions",
}


def load_records(data_path: Path, links_path: Path | None = None) -> list[dict[str, str]]:
    """Load the dataset's JSON-lines strings and align them with source URLs."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    links = _load_links(links_path)
    records: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            warnings.warn(f"Skipping malformed JSON record at line {line_number}: {exc.msg}", stacklevel=2)
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        records.append(
            {
                "record_id": f"record-{line_number:04d}",
                "text": _normalize(value),
                "source": links[len(records)] if len(records) < len(links) else data_path.name,
            }
        )
    return records


def chunk_records(records: list[dict[str, str]], chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= overlap < chunk_size")
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for page, record in enumerate(records, 1):
        text = record["text"].strip()
        if not text:
            continue
        title, body = _title_and_body(text)
        section = _section_from_source(record["source"], title)
        for piece in _paragraph_chunks(body, chunk_size, overlap):
            normalized = piece.strip()
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if not normalized or digest in seen:
                continue
            seen.add(digest)
            chunk_id = hashlib.sha256(
                f"{record['record_id']}|{section}|{normalized}".encode("utf-8")
            ).hexdigest()[:20]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": normalized,
                    "source": record["source"],
                    "document_id": record["record_id"],
                    "title": title,
                    "section": section,
                    "page": page,
                    "record_id": record["record_id"],
                }
            )
    return chunks


def dataset_fingerprint(chunks: list[Chunk], model_name: str) -> str:
    payload = model_name + "|" + "|".join(chunk["chunk_id"] for chunk in chunks)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_links(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _title_and_body(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0] if lines else "Untitled record"
    body_lines = [line for line in lines[1:] if line.lower() not in BOILERPLATE]
    return title, "\n".join(body_lines) or title


def _section_from_source(source: str, fallback: str) -> str:
    if source.startswith("http"):
        slug = source.rstrip("/").rsplit("/", 1)[-1]
        return slug.replace("-", " ").title() or fallback
    return fallback


def _paragraph_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack newline-delimited semantic units, splitting only units that exceed the limit."""
    units: list[str] = []
    for paragraph in (part.strip() for part in text.split("\n")):
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_window(paragraph, chunk_size, overlap))

    results: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = "\n".join([*current, unit])
        if current and len(candidate) > chunk_size:
            completed = "\n".join(current)
            results.append(completed)
            carry = _overlap_tail(completed, overlap)
            current = [carry, unit] if carry else [unit]
        else:
            current.append(unit)
    if current:
        results.append("\n".join(current))
    return results


def _window(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [text[start : start + size] for start in range(0, len(text), step)]


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap == 0:
        return ""
    tail = text[-overlap:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :] if first_space >= 0 else tail
