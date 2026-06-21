from pathlib import Path

import pytest

from school_rag.data import chunk_records, load_records


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_records_preserves_source_and_record_metadata():
    records = load_records(FIXTURES / "universities.jsonl", FIXTURES / "links.txt")
    assert len(records) == 2
    assert records[0]["record_id"] == "record-0001"
    assert records[0]["source"].endswith("/nile/about")


def test_loader_skips_empty_and_malformed_input():
    path = FIXTURES / "malformed.jsonl"
    with pytest.warns(UserWarning, match="line 2"):
        records = load_records(path)
    assert [item["text"] for item in records] == ["valid record"]


def test_chunking_preserves_boundaries_metadata_and_overlap():
    records = [{"record_id": "r1", "source": "https://example.edu/u/programs", "text": "University\n" + "A" * 90}]
    chunks = chunk_records(records, chunk_size=40, overlap=10)
    assert len(chunks) == 3
    assert chunks[0]["text"][-10:] == chunks[1]["text"][:10]
    assert all(chunk["section"] == "Programs" for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_records([], chunk_size=10, overlap=10)
