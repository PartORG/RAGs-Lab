"""Retrieve-and-Rerank: wide vector search for recall, then a cross-encoder for precision.

Vector search embeds the question and each chunk separately; a cross-encoder reads the pair
together, so it judges relevance far more precisely. The top 20 candidates go in, and only the
3 the cross-encoder rates highest reach the LLM. It cannot recover a chunk the vector search
never returned.
"""

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rags.naive_rag import NaiveRAG, Result, answer

if TYPE_CHECKING:  # importing it loads torch: several seconds, so the caller builds the model
    from sentence_transformers import CrossEncoder


def rerank(
    reranker: "CrossEncoder", question: str, docs: list[Document], k: int
) -> list[tuple[Document, float]]:
    """The k docs the cross-encoder rates most relevant to the question, best first."""
    scores = reranker.predict([(question, doc.page_content) for doc in docs])
    ranked = sorted(zip(docs, scores, strict=True), key=lambda t: t[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:k]]


class RerankRAG:
    """Reranks on top of Naive RAG's index, so both strategies search exactly the same chunks."""

    def __init__(self, index: NaiveRAG, reranker: "CrossEncoder", fetch_k: int = 20, k: int = 3):
        self.index, self.reranker, self.fetch_k, self.k = index, reranker, fetch_k, k

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        candidates = self.index.store.similarity_search_with_score(question, k=self.fetch_k)
        cosine = {doc.id: score for doc, score in candidates}
        best = rerank(self.reranker, question, [doc for doc, _ in candidates], self.k)
        result = answer(question, [(doc, cosine[doc.id]) for doc, _ in best], llm)
        result.rerank_scores = [score for _, score in best]
        return result
