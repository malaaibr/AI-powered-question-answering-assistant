from __future__ import annotations

from school_rag.graph import RAGWorkflow
from school_rag.prompts import GROUNDING_PROMPT, INSUFFICIENT_ANSWER

from conftest import retrieved_chunk


class FakeLLM:
    def __init__(self, fail_rewrite: bool = False):
        self.fail_rewrite = fail_rewrite
        self.generate_calls = 0
        self.last_prompt = ""

    def rewrite(self, query: str) -> str:
        if self.fail_rewrite:
            raise RuntimeError("offline")
        return f"clear {query}"

    def generate(self, prompt: str) -> str:
        self.generate_calls += 1
        self.last_prompt = prompt
        return "It was founded in 2018. [Source 1]"


class FakeStore:
    def __init__(self, results=None):
        self.results = results or []
        self.query = None
        self.top_k = None

    def search(self, query: str, top_k: int):
        self.query, self.top_k = query, top_k
        return sorted(self.results, key=lambda item: item["score"], reverse=True)[:top_k]


def test_rewrite_updates_state_and_retrieval_respects_top_k_order():
    store = FakeStore([retrieved_chunk(score=0.4), retrieved_chunk("other", score=0.8)])
    workflow = RAGWorkflow(FakeLLM(), store, top_k=1)
    result = workflow.run("when founded?")
    assert result["rewritten_query"] == "clear when founded?"
    assert store.query == result["rewritten_query"]
    assert store.top_k == 1
    assert result["retrieved_chunks"][0]["score"] == 0.8


def test_rewrite_failure_falls_back_to_original_query():
    result = RAGWorkflow(FakeLLM(fail_rewrite=True), FakeStore()).run("original")
    assert result["rewritten_query"] == "original"
    assert "rewrite failed" in result["error"].lower()


def test_no_results_skips_generation_and_returns_insufficient_answer():
    llm = FakeLLM()
    result = RAGWorkflow(llm, FakeStore()).run("unknown")
    assert result["answer"] == INSUFFICIENT_ANSWER
    assert result["sources"] == []
    assert llm.generate_calls == 0


def test_generation_prompt_is_grounded_and_sources_are_normalized():
    llm = FakeLLM()
    result = RAGWorkflow(llm, FakeStore([retrieved_chunk()])).run("When was it founded?")
    assert "[Source 1]" in llm.last_prompt
    assert "When was it founded?" in llm.last_prompt
    assert result["sources"][0]["source"].startswith("https://")
    assert "only using" in GROUNDING_PROMPT
    assert "Do not fabricate citations" in GROUNDING_PROMPT


def test_graph_has_required_node_order():
    graph = RAGWorkflow(FakeLLM(), FakeStore()).graph.get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("__start__", "rewrite") in edges
    assert ("rewrite", "retrieve") in edges
    assert ("retrieve", "generate") in edges
    assert ("generate", "__end__") in edges


def test_empty_query_flows_without_retrieval_or_generation():
    llm, store = FakeLLM(), FakeStore([retrieved_chunk()])
    result = RAGWorkflow(llm, store).run("  ")
    assert result["answer"] == INSUFFICIENT_ANSWER
    assert store.query is None
    assert llm.generate_calls == 0
