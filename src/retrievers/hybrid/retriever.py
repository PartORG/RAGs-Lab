"""Dense and BM25 together, merged with Reciprocal Rank Fusion.

Meaning and exact tokens are different failure modes, and fusing the two ranked lists covers
both. RRF needs no score calibration: it only reads the positions.
"""

from rags.hybrid_rag import BM25, rrf
from rags.naive_rag import NaiveRAG
from retrievers.common import Retriever, scored


def hybrid(index: NaiveRAG, fetch: int = 3) -> Retriever:
    def retrieve(question: str, k: int) -> list:
        wide = k * fetch
        dense = index.store.similarity_search(question, k=wide)
        sparse = BM25(index.chunks()).search(question, wide)
        return scored(index, question, rrf([dense, sparse])[:k])

    return retrieve
