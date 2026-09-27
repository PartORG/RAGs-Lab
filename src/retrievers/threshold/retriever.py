"""Similarity with a floor: return fewer chunks rather than weak ones.

Top-k always returns k chunks, even when the corpus holds nothing relevant — and the model then
answers from whatever came back. This drops everything below a cosine floor, so a question the
documents cannot answer arrives with little or no context, which is the honest input.
"""

from rags.naive_rag import NaiveRAG
from retrievers.common import Retriever

FLOOR = 0.5  # cosine; nomic-embed-text puts unrelated text around 0.3-0.45 on this corpus


def threshold(index: NaiveRAG, floor: float = FLOOR) -> Retriever:
    def retrieve(question: str, k: int) -> list:
        hits = index.store.similarity_search_with_score(question, k=k)
        return [(doc, score) for doc, score in hits if score >= floor]

    return retrieve
