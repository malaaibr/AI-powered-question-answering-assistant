from school_rag.store import ChromaVectorStore


class FakeCollection:
    def count(self):
        return 2

    def query(self, **kwargs):
        return {
            "ids": [["low", "high"]],
            "documents": [["low text", "high text"]],
            "metadatas": [[_metadata(), _metadata()]],
            "distances": [[0.7, 0.1]],
        }


class FakeEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0]


def _metadata():
    return {
        "source": "source",
        "document_id": "doc",
        "title": "title",
        "section": "section",
        "page": 1,
        "record_id": "record",
    }


def test_chroma_distance_is_converted_and_labeled_as_similarity():
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = FakeCollection()
    store.embedder = FakeEmbedder()
    results = store.search("query", 2)
    assert [item["chunk_id"] for item in results] == ["high", "low"]
    assert results[0]["score"] == 0.9
    assert results[0]["score_type"] == "cosine_similarity"
