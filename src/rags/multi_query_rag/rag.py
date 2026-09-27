"""Multi-Query RAG / RAG-Fusion: the LLM rewrites the question several ways, each rewrite is
searched, and the ranked lists are merged with Reciprocal Rank Fusion.

Chunks that keep turning up across phrasings float to the top, so one badly worded question no
longer decides what gets retrieved. Costs one extra LLM call per question.
"""

import re

from langchain_core.language_models import BaseChatModel

from rags.hybrid_rag import rrf
from rags.naive_rag import NaiveRAG, Result, answer, with_similarity

REWRITE_PROMPT = """Rewrite this question {n} different ways, using different words and angles.
Reply with one rewrite per line and nothing else.

Question: {question}"""


class MultiQueryRAG:
    def __init__(self, index: NaiveRAG, n: int = 3, per_query: int = 5, k: int = 4):
        self.index, self.n, self.per_query, self.k = index, n, per_query, k

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        reply = llm.invoke(REWRITE_PROMPT.format(n=self.n, question=question)).text
        # Models number their lines despite being asked not to: strip "1." / "-" prefixes.
        rewrites = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line) for line in reply.splitlines()]
        queries = [question, *[r.strip() for r in rewrites if r.strip()][: self.n]]
        rankings = [[doc for doc, _ in self.index.search(q, self.per_query)] for q in queries]
        fused = rrf(rankings)[: self.k]
        embeddings = self.index.store.embeddings
        result = answer(question, with_similarity(question, fused, embeddings), llm)
        result.trace = "Searched with:\n" + "\n".join(f"- {q}" for q in queries)
        return result
