"""What every retriever shares: the type it implements and where the index comes from."""

from typing import TYPE_CHECKING

from rags.naive_rag import Retriever, with_similarity

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from rags.naive_rag import NaiveRAG

__all__ = ["Retriever", "scored", "with_similarity"]


def scored(
    index: "NaiveRAG", question: str, docs: list["Document"]
) -> list[tuple["Document", float]]:
    """Attach each doc's cosine similarity to the question, for retrievers that rank by something
    else (keywords, diversity, a cross-encoder) but still have to report a comparable number."""
    return with_similarity(question, docs, index.store.embeddings)
