"""Maximal Marginal Relevance: relevant chunks that are not near-copies of each other.

Plain top-k happily returns the same paragraph four times when a document repeats itself. MMR
picks each next chunk for relevance minus similarity to what it already chose, trading a little
relevance for coverage.
"""

from rags.naive_rag import NaiveRAG
from retrievers.common import Retriever, scored

DIVERSITY = 0.5  # lambda_mult: 1.0 is pure relevance, 0.0 is pure diversity


def mmr(index: NaiveRAG, diversity: float = DIVERSITY, fetch: int = 5) -> Retriever:
    def retrieve(question: str, k: int) -> list:
        docs = index.store.max_marginal_relevance_search(
            question, k=k, fetch_k=k * fetch, lambda_mult=diversity
        )
        return scored(index, question, docs)

    return retrieve
