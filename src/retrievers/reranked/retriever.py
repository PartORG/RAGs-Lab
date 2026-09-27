"""Fetch wide, then let a cross-encoder choose: precision at the cost of a second model.

Embeddings score question and chunk separately; a cross-encoder reads them together and is far
more accurate. Retrieving 5k candidates and keeping the k it likes best puts that accuracy under
every strategy, not only Retrieve-and-Rerank.
"""

from collections.abc import Callable

from rags.naive_rag import NaiveRAG
from rags.retrieve_and_rerank import rerank
from retrievers.common import Retriever


def reranked(index: NaiveRAG, get_reranker: Callable, fetch: int = 5) -> Retriever:
    def retrieve(question: str, k: int) -> list:
        candidates = index.store.similarity_search_with_score(question, k=k * fetch)
        cosine = {doc.id: score for doc, score in candidates}
        best = rerank(get_reranker(), question, [doc for doc, _ in candidates], k)
        return [(doc, cosine[doc.id]) for doc, _ in best]

    return retrieve
