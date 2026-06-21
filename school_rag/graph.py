from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .models import RAGState, Source
from .prompts import INSUFFICIENT_ANSWER, generation_prompt
from .services import LLM
from .store import VectorStore


class RAGWorkflow:
    def __init__(self, llm: LLM, store: VectorStore, top_k: int = 5) -> None:
        self.llm = llm
        self.store = store
        self.top_k = top_k
        self.graph = self._build_graph()

    def rewrite(self, state: RAGState) -> dict[str, str]:
        query = state["original_query"].strip()
        if not query:
            return {"rewritten_query": "", "error": "The question cannot be empty."}
        try:
            rewritten = self.llm.rewrite(query).strip()
            return {"rewritten_query": rewritten or query}
        except Exception as exc:
            return {"rewritten_query": query, "error": f"Query rewrite failed; used original query: {exc}"}

    def retrieve(self, state: RAGState) -> dict[str, object]:
        query = state.get("rewritten_query", "").strip()
        return {"retrieved_chunks": self.store.search(query, self.top_k) if query else []}

    def generate(self, state: RAGState) -> dict[str, object]:
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {"answer": INSUFFICIENT_ANSWER, "sources": []}
        sources: list[Source] = []
        contexts: list[str] = []
        for number, chunk in enumerate(chunks, 1):
            sources.append(
                {
                    "number": number,
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "section": chunk["section"],
                    "page": chunk["page"],
                }
            )
            contexts.append(
                f"[Source {number}] title={chunk['title']}; section={chunk['section']}; "
                f"source={chunk['source']}\n{chunk['text']}"
            )
        prompt = generation_prompt(state["original_query"], state["rewritten_query"], contexts)
        return {"answer": self.llm.generate(prompt).strip(), "sources": sources}

    def _build_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("rewrite", self.rewrite)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("generate", self.generate)
        builder.add_edge(START, "rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def run(self, question: str) -> RAGState:
        return self.graph.invoke({"original_query": question})
