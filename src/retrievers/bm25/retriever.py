"""BM25 keyword search: no embeddings at all, just term statistics.

The classic lexical ranker. It cannot match a paraphrase, but it never misses an exact token —
an error code, a function name, a part number — which is where dense vectors are weakest.
"""

from rags.hybrid_rag import BM25
from rags.naive_rag import NaiveRAG
from retrievers.common import Retriever, scored


def bm25(index: NaiveRAG) -> Retriever:
    def retrieve(question: str, k: int) -> list:
        # ponytail: rebuilt per question (~50 ms for 600 chunks); cache it if the corpus grows.
        docs = BM25(index.chunks()).search(question, k)
        return scored(index, question, docs)

    return retrieve
