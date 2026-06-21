import logging
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

import schematic_diff_agent as module
from schematic_diff_agent import INSUFFICIENT_ANSWER, SchematicDiffAgent


class FakeModel:
    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("can" in lowered or "flex" in lowered),
                float("qspi" in lowered or "flash" in lowered),
                float("new" in lowered),
            ])
        return np.asarray(vectors, dtype="float32")

    def get_sentence_embedding_dimension(self):
        return 3


@pytest.fixture
def agent():
    chroma_client = MagicMock()
    chroma_client.get_or_create_collection.return_value = MagicMock()
    with patch.object(module, "SentenceTransformer", return_value=FakeModel()), \
            patch.object(module.chromadb, "PersistentClient", return_value=chroma_client), \
            patch.object(module.os, "makedirs"):
        return SchematicDiffAgent("work", event_logger=logging.getLogger("test"))


def test_tokenize_splits_alphanumeric_underscore_and_lowercases():
    assert SchematicDiffAgent._tokenize("FlexCAN0-TXD PB_01 / U17!") == [
        "flexcan0", "txd", "pb_01", "u17"
    ]


def test_get_chunks_from_two_page_pdf_preserves_metadata(agent):
    pages = []
    for text in ("A" * 20, "B" * 11):
        page = MagicMock()
        page.get_text.return_value = text
        pages.append(page)
    document = MagicMock()
    document.__iter__.return_value = pages
    with patch.object(module.fitz, "open", return_value=document):
        chunks, metadata = agent._get_chunks_from_pdf("board.pdf", "board.pdf", "OLD", 10, 2)
    assert len(chunks) == 5
    assert metadata[0] == {"version": "OLD", "source": "board.pdf", "page": 1}
    assert metadata[-1]["page"] == 2
    document.close.assert_called_once()


def test_hybrid_rrf_is_sorted_and_version_filtered(agent):
    chunks = ["FlexCAN TXD OLD", "QSPI flash OLD", "FlexCAN RXD NEW",
              "QSPI data NEW", "unrelated power OLD"]
    agent.chunks = chunks
    agent.metadata = [
        {"version": "OLD" if "OLD" in text else "NEW", "source": "x.pdf", "page": i + 1}
        for i, text in enumerate(chunks)
    ]
    vectors = agent._embed(chunks)
    agent.index = faiss.IndexFlatIP(3)
    agent.index.add(vectors)
    agent.bm25 = BM25Okapi([agent._tokenize(text) for text in chunks])
    results = agent.hybrid_search("FlexCAN NEW", k=3)
    assert [score for score, _, _ in results] == sorted(
        [score for score, _, _ in results], reverse=True
    )
    assert all(meta["score_type"] == "rrf" for _, _, meta in results)
    assert all(isinstance(meta["dense_similarity"], float) for _, _, meta in results)
    assert all(isinstance(meta["bm25_score"], float) for _, _, meta in results)
    filtered = agent.hybrid_search("FlexCAN", k=5, version="NEW")
    assert filtered
    assert all(meta["version"] == "NEW" for _, _, meta in filtered)


def test_format_context_matches_required_source_tag():
    text = SchematicDiffAgent._format_context([
        (0.03, "TXD -> U17", {"version": "OLD", "source": "old.pdf", "page": 22})
    ])
    assert text == "[VERSION: OLD, SOURCE: old.pdf, PAGE: 22]\nCONTENT: TXD -> U17"


def test_run_openai_passes_prompt_and_model(agent):
    client = MagicMock()
    client.responses.create.return_value.output_text = "answer"
    with patch.dict(module.os.environ, {"OPENAI_API_KEY": "test-key"}), \
            patch.object(module, "OpenAI", return_value=client):
        answer = agent._run_openai("prompt", model="gpt-test")
    assert answer == "answer"
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["input"][0]["content"][0] == {"type": "input_text", "text": "prompt"}


def test_run_openai_requires_api_key(agent):
    with patch.dict(module.os.environ, {}, clear=True), \
            pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        agent._run_openai("prompt")


def test_load_index_dimension_mismatch_has_rebuild_instruction(agent):
    wrong = faiss.IndexFlatIP(5)
    with patch.object(module.os.path, "exists", return_value=True), \
            patch.object(module.faiss, "read_index", return_value=wrong), \
            pytest.raises(ValueError, match="dimension mismatch") as error:
        agent._load_index()
    assert "Delete the _index folder" in str(error.value)


def test_markdown_diff_tables_export_mapping():
    report = """## Pin / Port Changes (IOMUX impact)
| interface_instance | signal_role | OLD soc_port | NEW soc_port | software action |
|---|---|---|---|---|
| FlexCAN0 | TXD | PB_01 | PC_02 | update IOMUX |
"""
    tables = SchematicDiffAgent._markdown_tables(report)
    row = tables["pin / port changes (iomux impact)"][0][0]
    assert row["old soc_port"] == "PB_01"


def test_chat_graph_order_and_end_to_end_grounded_state(agent):
    agent._run_openai = MagicMock(side_effect=[
        "FlexCAN0 TXD connection NEW",
        "FlexCAN0 TXD is connected on PB_01. [VERSION: NEW, SOURCE: new.pdf, PAGE: 7]",
    ])
    agent.hybrid_search = MagicMock(return_value=[
        (0.032, "FlexCAN0 TXD PB_01 connects to U17",
         {"version": "NEW", "source": "new.pdf", "page": 7})
    ])
    result = agent.chat("Where is CAN TX?", version="NEW", with_images=False)
    assert result["rewritten_query"] == "FlexCAN0 TXD connection NEW"
    assert result["answer"].endswith("PAGE: 7]")
    assert result["sources"] == [{"version": "NEW", "source": "new.pdf", "page": 7}]
    edges = {(edge.source, edge.target) for edge in agent.graph.get_graph().edges}
    assert [("__start__", "rewrite"), ("rewrite", "retrieve"),
            ("retrieve", "generate"), ("generate", "__end__")] == [
        edge for edge in [("__start__", "rewrite"), ("rewrite", "retrieve"),
                          ("retrieve", "generate"), ("generate", "__end__")] if edge in edges
    ]


def test_chat_rewrite_failure_falls_back_and_no_results_skips_generation(agent):
    agent._run_openai = MagicMock(side_effect=RuntimeError("offline"))
    agent.hybrid_search = MagicMock(return_value=[])
    result = agent.chat("What changed?", with_images=False)
    assert result["rewritten_query"] == "What changed?"
    assert result["answer"] == INSUFFICIENT_ANSWER
    assert result["sources"] == []
    agent._run_openai.assert_called_once()
