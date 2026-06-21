from __future__ import annotations

import argparse

from .config import Settings
from .factory import build_workflow


def _source_location(item: dict) -> str:
    return f"{item['title']} — {item['section']} (record {item['page']})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask grounded questions about Egyptian private universities")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-debug", action="store_true", help="hide retrieved chunk previews")
    args = parser.parse_args()
    settings = Settings.from_env()
    top_k = args.top_k if args.top_k is not None else settings.top_k
    if top_k <= 0:
        parser.error("--top-k must be greater than zero")
    workflow = build_workflow(Settings(**{**settings.__dict__, "top_k": top_k}))
    result = workflow.run(args.question)

    print(f"Original query: {result['original_query']}")
    print(f"Rewritten query: {result['rewritten_query']}")
    if result.get("error"):
        print(f"Note: {result['error']}")
    if not args.no_debug:
        print("\nRetrieved chunks:")
        for rank, chunk in enumerate(result.get("retrieved_chunks", []), 1):
            preview = chunk["text"].replace("\n", " ")[:220]
            print(f"{rank}. {chunk['score']:.4f} {chunk['score_type']} | {_source_location(chunk)}")
            print(f"   {preview}")
    print(f"\nAnswer:\n{result['answer']}")
    print("\nSources:")
    for source in result.get("sources", []):
        print(f"[Source {source['number']}] {_source_location(source)} | {source['source']}")


if __name__ == "__main__":
    main()
