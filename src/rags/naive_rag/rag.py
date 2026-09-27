"""Naive RAG: chunk -> embed -> top-k cosine search -> stuff the prompt -> answer.

The baseline every other strategy in this lab has to beat. After answering, the same model grades
its own answer so the UI can show how trustworthy the result looks. Its chunk index is shared:
the other strategies search it, or build their own index from its chunks.
"""

import logging
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field
from pypdf import PdfReader

from chunking_strategies.common import Chunker
from chunking_strategies.recursive import recursive
from storage import Saved, saved_of

log = logging.getLogger(__name__)

ANSWER_PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the documents."

Context:
{context}

Question: {question}"""

GRADE_PROMPT = """You are grading an answer produced by a retrieval-augmented system.

Context the system retrieved:
{context}

Question: {question}

Answer: {answer}

Reply in JSON with:
- reason: one or two sentences on whether the answer is supported by the context and whether it
  answers the question.
- groundedness (1-5): 5 = every claim in the answer is stated in the context, 1 = most of it is
  not. An answer that refuses claims nothing, so it scores 5.
- relevance (1-5): 5 = fully answers the question, 1 = does not answer it.

An answer that says the information is not there is a refusal, however it is worded ("I don't
know", "the context does not contain the answer", "the documents do not say"): score every
refusal groundedness 5 and relevance 1."""


class Grade(BaseModel):
    # Ollama uses the schema only to constrain the output; the rubric must live in GRADE_PROMPT.
    # reason comes first so the model reasons before it commits to the numbers.
    reason: str
    groundedness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)


# How candidates are fetched from the index: question and k in, chunks with their cosine
# similarity out. src/retrievers/ holds the implementations; None means plain cosine top-k.
Retriever = Callable[[str, int], list[tuple[Document, float]]]


@dataclass
class Result:
    answer: str
    chunks: list[tuple[Document, float]]  # retrieved chunk, cosine similarity to the question
    grade: Grade | None  # None when the model could not produce a valid grade
    grade_error: str = ""
    rerank_scores: list[float] | None = None  # cross-encoder score per chunk, when reranked
    trace: str = ""  # what the strategy did before retrieval (rewrites, drafts, ...), for the UI


def load(path: Path) -> list[Document]:
    """One Document per PDF page, or one for a whole text file."""
    if path.suffix.lower() == ".pdf":
        return [
            Document(page.extract_text() or "", metadata={"source": path.name, "page": i})
            for i, page in enumerate(PdfReader(path).pages, start=1)
        ]
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(text, metadata={"source": path.name})]


def with_similarity(
    question: str, docs: list[Document], embeddings: Embeddings
) -> list[tuple[Document, float]]:
    """Pair each doc with its cosine similarity to the question, so strategies that do not rank by
    it (fusion, rewrites, parents, graph facts) are still measured the same way in the UI."""
    if not docs:
        return []
    q = np.array(embeddings.embed_query(question))
    d = np.array(embeddings.embed_documents([doc.page_content for doc in docs]))
    sims = d @ q / (np.linalg.norm(d, axis=1) * np.linalg.norm(q))
    return list(zip(docs, sims.tolist(), strict=True))


def context_of(chunks: list[tuple[Document, float]]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc, _ in chunks)


def answer(question: str, chunks: list[tuple[Document, float]], llm: BaseChatModel) -> Result:
    """Answer from the chunks, then have the same model grade that answer."""
    reply = llm.invoke(ANSWER_PROMPT.format(context=context_of(chunks), question=question)).text
    return graded(question, chunks, reply, llm)


def graded(
    question: str, chunks: list[tuple[Document, float]], reply: str, llm: BaseChatModel
) -> Result:
    """A reply, however it was generated, graded by the model against the chunks it was given."""
    context = context_of(chunks)
    try:
        grade = llm.with_structured_output(Grade).invoke(
            GRADE_PROMPT.format(context=context, question=question, answer=reply)
        )
    except Exception as e:  # model output is untrusted: bad JSON, out-of-range scores, ...
        log.warning("The answer could not be graded: %r", e)
        return Result(reply, chunks, None, f"{type(e).__name__}: {e}")
    return Result(reply, chunks, grade)


def load_store(saved: Saved, embeddings: Embeddings) -> InMemoryVectorStore:
    """A saved vector store, or an empty one. Its records are plain dicts (id, vector, text,
    metadata), so they are saved as they are."""
    store = InMemoryVectorStore(embeddings)
    store.store = saved.load() or {}
    return store


def save_store(store: InMemoryVectorStore, saved: Saved) -> None:
    saved.save(store.store)


def remove_everywhere(owner: str, name: str) -> int:
    """Drop a file's chunks from every chunking x embedding index of this user, without opening
    them as NaiveRAGs, which would load each embedding model (~2 GB for bge-m3) only to compute
    nothing. Returns the chunks removed."""
    removed = 0
    for saved in saved_of(owner, "naive_rag"):
        records = saved.load()
        keep = {i: r for i, r in records.items() if r["metadata"]["source"] != name}
        if len(keep) < len(records):
            saved.save(keep)
            removed += len(records) - len(keep)
    return removed


class NaiveRAG:
    # ponytail: brute-force in-memory cosine search, saved whole as one SQLite row per user and
    # index; move to sqlite-vec or Qdrant when a corpus outgrows RAM or saving on every upload
    # gets slow.
    def __init__(
        self,
        embeddings: Embeddings,
        saved: Saved,
        k: int = 4,
        chunker: Chunker | None = None,
        retriever: Retriever | None = None,
    ):
        self.saved, self.k = saved, k
        self.chunker = chunker or recursive()  # how documents are cut; see chunking_strategies
        self.retriever = retriever  # how candidates are found; see retrievers
        self.store = load_store(saved, embeddings)
        # Two tabs of one user share this index (the page caches it): one change saves at a time.
        self.lock = threading.Lock()

    def sources(self) -> Counter[str]:
        """Indexed file name -> number of chunks."""
        return Counter(r["metadata"]["source"] for r in self.store.store.values())

    def chunks(self) -> list[Document]:
        """Every indexed chunk, in indexing order, with its id."""
        records = self.store.store.items()
        return [Document(r["text"], id=i, metadata=dict(r["metadata"])) for i, r in records]

    def add_file(self, path: Path) -> int:
        """(Re)index one file, replacing any earlier version with the same name. Returns chunks."""
        chunks = [c for c in self.chunker(load(path)) if c.page_content.strip()]
        with self.lock:
            self._forget(path.name)
            if chunks:
                self.store.add_documents(chunks)
            save_store(self.store, self.saved)
        return len(chunks)

    def remove_file(self, name: str) -> int:
        """Drop a file's chunks. The strategies with their own index notice on their next look,
        because the chunk ids they were built from are gone. Returns the chunks removed."""
        with self.lock:
            removed = self._forget(name)
            save_store(self.store, self.saved)
        return removed

    def _forget(self, name: str) -> int:
        ids = [i for i, r in self.store.store.items() if r["metadata"]["source"] == name]
        self.store.delete(ids)
        return len(ids)

    def search(self, question: str, k: int | None = None) -> list[tuple[Document, float]]:
        """The one place a strategy fetches candidates, so the chosen retriever applies to all of
        them. Strategies that hard-wire their own retrieval (Hybrid, Rerank, ...) do not use it."""
        k = self.k if k is None else k
        if self.retriever:
            return self.retriever(question, k)
        return self.store.similarity_search_with_score(question, k=k)

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        return answer(question, self.search(question), llm)
