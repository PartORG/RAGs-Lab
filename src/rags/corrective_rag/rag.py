"""Corrective RAG (CRAG): grade every retrieved chunk and never generate from garbage.

Chunks that fail the relevance check are dropped. When none pass, the pipeline corrects itself:
the LLM rewrites the question as a search query, which is searched again, wider, with dense +
keyword (BM25) search, and graded again. When that fails too it says so, instead of improvising.

The catalogue's fallback is a web search. That is off by default here, because this lab keeps
everything on the machine; switched on, only the rewritten question is sent to DuckDuckGo, never
the documents.
"""

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rags.hybrid_rag import BM25, rrf
from rags.naive_rag import NaiveRAG, Result, answer, graded, with_similarity

log = logging.getLogger(__name__)

RELEVANT_PROMPT = """Question: {question}

Chunk:
{chunk}

Does this chunk contain information that helps answer the question? Answer YES or NO only."""

REWRITE_PROMPT = """Rewrite this question as a query for a document search engine: keep the key
names and terms, drop filler words. Reply with the query only.

Question: {question}"""

NOT_FOUND = "I don't know based on the documents."


def web_snippets(query: str, n: int = 3) -> list[Document]:
    """DuckDuckGo results as chunks. The one place this lab reaches the network."""
    from ddgs import DDGS  # only imported when the fallback is switched on

    hits = DDGS().text(query, max_results=n)
    return [
        Document(f"{hit['title']}\n{hit['body']}", metadata={"source": f"web: {hit['href']}"})
        for hit in hits
    ]


def is_yes(llm: BaseChatModel, prompt: str) -> bool:
    return llm.invoke(prompt).text.strip().upper().startswith("YES")


class CorrectiveRAG:
    def __init__(self, index: NaiveRAG, k: int = 4, fallback_k: int = 8, web: bool = False):
        self.index, self.k, self.fallback_k, self.web = index, k, fallback_k, web

    def _relevant(
        self, question: str, chunks: list[tuple[Document, float]], llm: BaseChatModel
    ) -> list[tuple[Document, float]]:
        return [
            (doc, score)
            for doc, score in chunks
            if is_yes(llm, RELEVANT_PROMPT.format(question=question, chunk=doc.page_content))
        ]

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        hits = self.index.search(question, self.k)
        relevant = self._relevant(question, hits, llm)
        trace = [f"Graded {len(hits)} retrieved chunks: {len(relevant)} relevant."]
        if not relevant:
            query = llm.invoke(REWRITE_PROMPT.format(question=question)).text.strip()
            dense = self.index.store.similarity_search(query, k=self.fallback_k)
            sparse = BM25(self.index.chunks()).search(query, self.fallback_k)
            fused = rrf([dense, sparse])[: self.fallback_k]
            candidates = with_similarity(question, fused, self.index.store.embeddings)
            relevant = self._relevant(question, candidates, llm)
            trace.append(
                f"Rewrote the question as the query: {query}\n"
                f"Searched again (dense + keyword): {len(relevant)} of {len(candidates)} relevant."
            )
        if not relevant and self.web:
            try:
                candidates = with_similarity(
                    question, web_snippets(query), self.index.store.embeddings
                )
            except Exception as e:  # offline, rate-limited, blocked: stay with the documents
                log.warning("CRAG web fallback failed: %r", e)
                candidates = []
                trace.append(f"The web fallback failed ({type(e).__name__}).")
            if candidates:
                relevant = self._relevant(question, candidates, llm)
                trace.append(
                    f"Asked the web (DuckDuckGo) for: {query}\n"
                    f"{len(relevant)} of {len(candidates)} results relevant."
                )
        if relevant:
            result = answer(question, relevant, llm)
        else:
            trace.append("Nothing relevant found: said so instead of answering from bad context.")
            result = graded(question, [], NOT_FOUND, llm)
        result.trace = "\n".join(trace)
        return result
