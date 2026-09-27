"""How candidates are found — the step between the index and every strategy.

A strategy decides what to do with chunks; the retriever decides which chunks it gets. Most
strategies here just search the index, and they all go through `NaiveRAG.search`, so the choice
made in the sidebar applies to all of them at once. The exceptions are the strategies whose whole
point is their own retrieval — Hybrid RAG, Retrieve-and-Rerank, Contextual RAG, Hierarchical RAG,
GraphRAG, Branched and Self-Query — which keep doing what they were built to do.

Unlike a chunking or an embedding, a retriever changes nothing on disk: it is a query-time
choice, so switching costs no re-indexing.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rags.naive_rag import NaiveRAG, Retriever
from retrievers.bm25.retriever import bm25 as _bm25
from retrievers.hybrid.retriever import hybrid as _hybrid
from retrievers.mmr.retriever import mmr as _mmr
from retrievers.reranked.retriever import reranked as _reranked
from retrievers.similarity.retriever import similarity as _similarity
from retrievers.threshold.retriever import threshold as _threshold

__all__ = ["DEFAULT", "RETRIEVERS", "Retrieval", "Retriever"]


@dataclass(frozen=True)
class Retrieval:
    label: str
    describe: str
    # get_reranker is called only by the retriever that needs it, so picking any other one never
    # loads the cross-encoder.
    build: Callable[[NaiveRAG, Callable], Retriever]


RETRIEVERS: dict[str, Retrieval] = {
    "similarity": Retrieval(
        "Similarity (top-k)",
        "The k nearest chunks by cosine. Always returns k, however weak the best one is.",
        lambda index, get_reranker: _similarity(index),
    ),
    "mmr": Retrieval(
        "MMR (diverse)",
        "Relevant but not repetitive: each pick is penalised for looking like the last.",
        lambda index, get_reranker: _mmr(index),
    ),
    "threshold": Retrieval(
        "Similarity with a floor",
        "Drops anything below 0.5 cosine, so a question with no answer gets no context.",
        lambda index, get_reranker: _threshold(index),
    ),
    "bm25": Retrieval(
        "BM25 (keywords)",
        "Lexical only: exact tokens, no embeddings. Misses paraphrases, never misses a code.",
        lambda index, get_reranker: _bm25(index),
    ),
    "hybrid": Retrieval(
        "Hybrid (dense + BM25)",
        "Both lists, merged with Reciprocal Rank Fusion. The usual production default.",
        lambda index, get_reranker: _hybrid(index),
    ),
    "reranked": Retrieval(
        "Reranked (cross-encoder)",
        "Fetches 5x, then a cross-encoder keeps the best k. Slowest, most precise.",
        lambda index, get_reranker: _reranked(index, get_reranker),
    ),
}

DEFAULT = "similarity"
