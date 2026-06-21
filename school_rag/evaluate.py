from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .factory import build_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and record the five-question evaluation set")
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    workflow = build_workflow(Settings.from_env())
    results = []
    for item in questions:
        state = workflow.run(item["query"])
        results.append(
            {
                **item,
                "rewritten_query": state["rewritten_query"],
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "score": chunk["score"],
                        "score_type": chunk["score_type"],
                        "source": chunk["source"],
                        "section": chunk["section"],
                        "preview": chunk["text"][:300],
                    }
                    for rank, chunk in enumerate(state.get("retrieved_chunks", []), 1)
                ],
                "answer": state["answer"],
                "sources": state["sources"],
                "error": state.get("error"),
            }
        )
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} evaluation records to {args.output}")


if __name__ == "__main__":
    main()
