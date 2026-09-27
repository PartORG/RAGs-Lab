"""Dense similarity search: the k nearest chunks by cosine. The default, and the baseline."""

from rags.naive_rag import NaiveRAG
from retrievers.common import Retriever


def similarity(index: NaiveRAG) -> Retriever:
    def retrieve(question: str, k: int) -> list:
        return index.store.similarity_search_with_score(question, k=k)

    return retrieve
