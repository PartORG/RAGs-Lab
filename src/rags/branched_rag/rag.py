"""Branched / Multi-Source RAG: ask every source at once, always, and merge.

Routing is static, not agentic: the same question always consults the same branches, which makes
it predictable and as slow as its slowest branch rather than the sum. This lab's branches are the
text chunks by meaning, the same chunks by keyword, and whichever other indexes have been built
(RAPTOR summaries, figure captions). Their rankings are merged with Reciprocal Rank Fusion.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rags.hybrid_rag import BM25, rrf
from rags.naive_rag import NaiveRAG, Result, answer, with_similarity

Branch = tuple[str, Callable[[str, int], list[Document]]]


class BranchedRAG:
    def __init__(
        self, index: NaiveRAG, extra: list[Branch] | None = None, per_branch: int = 3, k: int = 4
    ):
        self.index, self.per_branch, self.k = index, per_branch, k
        self.branches: list[Branch] = [
            ("text chunks", lambda q, n: index.store.similarity_search(q, k=n)),
            ("keywords (BM25)", lambda q, n: BM25(index.chunks()).search(q, n)),
            *(extra or []),
        ]

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        with ThreadPoolExecutor(max_workers=len(self.branches)) as pool:
            hits = list(pool.map(lambda b: b[1](question, self.per_branch), self.branches))
        fused = rrf(hits)[: self.k]
        chunks = with_similarity(question, fused, self.index.store.embeddings)
        result = answer(question, chunks, llm)
        result.trace = "Asked every source at once:\n" + "\n".join(
            f"- {name}: {len(found)} passages"
            for (name, _), found in zip(self.branches, hits, strict=True)
        )
        return result
