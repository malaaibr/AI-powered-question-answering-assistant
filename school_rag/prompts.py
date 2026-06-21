REWRITE_PROMPT = """You are a search-query rewriting component. Rewrite the user's query into a clear, specific query optimized for semantic retrieval. Preserve the original intent, names, identifiers, numbers, and technical terminology. Translate to English when necessary. Do not answer the question and do not introduce new facts. Return only the rewritten query."""

GROUNDING_PROMPT = """You are a grounded question-answering assistant. Answer only using the supplied context. Do not use unsupported outside knowledge. If the context does not contain enough information, explicitly state that the available sources are insufficient. Cite supporting context blocks using [Source N]. Do not fabricate citations. Keep the answer concise and accurate."""

INSUFFICIENT_ANSWER = "The available sources are insufficient to answer this question."


def generation_prompt(original_query: str, rewritten_query: str, contexts: list[str]) -> str:
    blocks = "\n\n".join(contexts)
    return (
        f"Original question: {original_query}\n"
        f"Retrieval query: {rewritten_query}\n\n"
        f"Retrieved context:\n{blocks}\n\n"
        "Answer the original question using only the numbered context above."
    )
