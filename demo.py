#!/usr/bin/env python3
"""One-command end-to-end demo for the Schematic IRAG question-answering path."""

import argparse
import logging

from schematic_diff_agent import SchematicDiffAgent


def main():
    parser = argparse.ArgumentParser(
        description="Ingest OLD/NEW PDFs, build the hybrid index, and ask one grounded question."
    )
    parser.add_argument("--old", required=True, help="Path to the OLD schematic PDF.")
    parser.add_argument("--new", required=True, help="Path to the NEW schematic PDF.")
    parser.add_argument(
        "--question",
        default="What changed between OLD and NEW that may require software updates?",
    )
    parser.add_argument("--workdir", default="./work")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    agent = SchematicDiffAgent(args.workdir, args.embedding_model)

    old_summary = agent.ingest("OLD", args.old, args.chunk_size, args.overlap, args.dpi)
    new_summary = agent.ingest("NEW", args.new, args.chunk_size, args.overlap, args.dpi)
    index_summary = agent.build_index()
    result = agent.chat(
        args.question,
        k=args.k,
        model=args.openai_model,
        with_images=not args.no_images,
    )

    print("\n=== INGESTION ===")
    print(f"OLD: {old_summary['chunks']} chunks, {old_summary['images']} pages")
    print(f"NEW: {new_summary['chunks']} chunks, {new_summary['images']} pages")
    print(f"Index: {index_summary['chunks']} chunks, dimension {index_summary['dimension']}")
    print("\n=== QUESTION ===")
    print(result["original_query"])
    print("\n=== REWRITTEN QUERY ===")
    print(result["rewritten_query"])
    print("\n=== RETRIEVED CHUNKS ===")
    for rank, chunk in enumerate(result.get("retrieved_chunks", []), 1):
        cosine = chunk.get("dense_similarity")
        cosine_text = f"{cosine:.6f}" if cosine is not None else "n/a"
        preview = chunk["text"].replace("\n", " ")[:240]
        print(
            f"{rank}. RRF={chunk['score']:.6f}; cosine={cosine_text}; "
            f"BM25={chunk.get('bm25_score', 0.0):.6f}; "
            f"{chunk['version']} {chunk['source']} page {chunk['page']}"
        )
        print(f"   {preview}")
    print("\n=== GROUNDED ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for source in result.get("sources", []):
        print(f"{source['version']} | {source['source']} | page {source['page']}")


if __name__ == "__main__":
    main()
