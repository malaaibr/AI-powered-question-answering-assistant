from pathlib import Path

from school_rag.data import chunk_records, load_records
from school_rag.graph import RAGWorkflow


class KeywordStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k):
        terms = set(query.lower().replace("?", "").split())
        scored = []
        for chunk in self.chunks:
            score = len(terms & set(chunk["text"].lower().split())) / max(len(terms), 1)
            if score:
                scored.append({**chunk, "score": score, "score_type": "token_overlap"})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


class DeterministicLLM:
    def rewrite(self, query):
        return "Nile Technical University founded year"

    def generate(self, prompt):
        assert "founded in 2018" in prompt
        return "Nile Technical University was founded in 2018. [Source 1]"


def test_end_to_end_graph_retrieves_expected_source_and_cites_it():
    fixture = Path(__file__).parent / "fixtures"
    chunks = chunk_records(load_records(fixture / "universities.jsonl", fixture / "links.txt"), 300, 30)
    result = RAGWorkflow(DeterministicLLM(), KeywordStore(chunks), top_k=2).run("When was Nile founded?")
    assert result["answer"].endswith("[Source 1]")
    assert result["sources"][0]["source"].endswith("/nile/about")
    assert result["retrieved_chunks"][0]["score_type"] == "token_overlap"
