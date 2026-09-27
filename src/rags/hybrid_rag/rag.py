"""Hybrid RAG: dense (embedding) search + sparse (BM25 keyword) search, fused with RRF.

Embeddings understand meaning but blur exact tokens (function names, error codes, IDs); BM25 nails
exact tokens but misses paraphrases. Reciprocal Rank Fusion merges the two ranked lists without
having to calibrate their very different scores against each other.
"""

import math
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rags.naive_rag import NaiveRAG, Result, answer, with_similarity


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    """Okapi BM25 over a fixed list of docs; a dozen lines instead of a dependency."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tfs = [Counter(tokens(d.page_content)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / max(len(docs), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def score(self, query: list[str], tf: Counter[str], length: int) -> float:
        norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
        return sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)

    def search(self, query: str, k: int) -> list[Document]:
        q = tokens(query)
        scores = [self.score(q, tf, n) for tf, n in zip(self.tfs, self.lens, strict=True)]
        best = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [self.docs[i] for i in best if scores[i] > 0]


def rrf(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: docs ranked high in many lists win; scores never need comparing."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in rankings:
        for rank, doc in enumerate(ranked):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            docs[doc.id] = doc
    return [docs[i] for i in sorted(scores, key=scores.__getitem__, reverse=True)]


class HybridRAG:
    def __init__(self, index: NaiveRAG, fetch_k: int = 10, k: int = 4):
        self.index, self.fetch_k, self.k = index, fetch_k, k

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        # ponytail: BM25 is rebuilt from all chunks on every question (~50 ms for 600 chunks);
        # keep it cached next to the index once the corpus reaches tens of thousands of chunks.
        dense = self.index.store.similarity_search(question, k=self.fetch_k)
        sparse = BM25(self.index.chunks()).search(question, self.fetch_k)
        fused = rrf([dense, sparse])[: self.k]
        embeddings = self.index.store.embeddings
        return answer(question, with_similarity(question, fused, embeddings), llm)
