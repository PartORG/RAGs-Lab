"""Contextual RAG (Anthropic's contextual retrieval): an LLM-written header situates each chunk.

"The ball bounces off the bat" says nothing about which game. At build time the LLM reads the
start of the chunk's document (title, contents, intro) plus the chunk, and writes 1-2 sentences
placing it; header + chunk is what gets embedded and BM25-indexed. Questions then take the
Hybrid (#3) + Rerank (#2) path over the contextualized chunks.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rags.hybrid_rag import BM25, rrf
from rags.naive_rag import NaiveRAG, Result, answer, load_store, save_store, with_similarity
from rags.retrieve_and_rerank import rerank
from storage import Saved

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

CONTEXT_PROMPT = """<document>
{document}
</document>
Here is a chunk{where} of this document:
<chunk>
{chunk}
</chunk>
In 1-2 sentences, state where this chunk sits in the document and what it covers, to improve
search retrieval. Reply with only those sentences."""


def situate(llm: BaseChatModel, document: str, chunk: Document) -> Document:
    """The chunk with an LLM-written header placing it in its document."""
    page = chunk.metadata.get("page")
    where = f" from page {page}" if page else ""
    prompt = CONTEXT_PROMPT.format(document=document, where=where, chunk=chunk.page_content)
    header = llm.invoke(prompt).text.strip()
    return Document(f"{header}\n\n{chunk.page_content}", id=chunk.id, metadata=chunk.metadata)


class ContextualRAG:
    seconds_per_item = 1.3  # one short LLM call per chunk, measured with qwen3:8b

    def __init__(
        self,
        index: NaiveRAG,
        saved: Saved,
        reranker: "CrossEncoder",
        fetch_k: int = 20,
        k: int = 3,
    ):
        self.index, self.saved, self.reranker = index, saved, reranker
        self.fetch_k, self.k = fetch_k, k
        emb = index.store.embeddings
        self.store = load_store(saved, emb)

    def _todo(self) -> list[Document]:
        """Chunks without a header yet; drops chunks a re-upload replaced."""
        chunks = self.index.store.store
        orphans = [i for i in self.store.store if i not in chunks]
        if orphans:
            self.store.delete(orphans)
            save_store(self.store, self.saved)
        return [c for c in self.index.chunks() if c.id not in self.store.store]

    def pending(self) -> int:
        return len(self._todo())

    def build(self, llm: BaseChatModel, progress: Callable[[int, int], None]) -> None:
        todo = self._todo()
        texts = defaultdict(list)
        for chunk in self.index.chunks():
            texts[chunk.metadata["source"]].append(chunk.page_content)
        # ponytail: the first 8000 characters stand in for the whole document, which would not fit
        # Ollama's default 4096-token context; raise num_ctx and this limit if the GPU allows.
        # Being the same prefix for every chunk of a file, Ollama reuses it from its cache.
        starts = {source: "\n".join(parts)[:8000] for source, parts in texts.items()}
        for i in range(0, len(todo), 20):  # save every 20 chunks, so an interrupted build resumes
            batch = todo[i : i + 20]
            docs = [situate(llm, starts[c.metadata["source"]], c) for c in batch]
            self.store.add_documents(docs)
            save_store(self.store, self.saved)
            progress(i + len(batch), len(todo))

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        records = self.store.store.items()
        docs = [Document(r["text"], id=i, metadata=dict(r["metadata"])) for i, r in records]
        dense = self.store.similarity_search(question, k=self.fetch_k)
        sparse = BM25(docs).search(question, self.fetch_k)
        candidates = rrf([dense, sparse])[: self.fetch_k]
        best = rerank(self.reranker, question, candidates, self.k)
        embeddings = self.store.embeddings
        result = answer(question, with_similarity(question, [d for d, _ in best], embeddings), llm)
        result.rerank_scores = [score for _, score in best]
        return result
