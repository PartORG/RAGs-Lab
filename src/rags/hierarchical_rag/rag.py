"""Hierarchical RAG (parent-child, small-to-big): match on small child chunks, read their parents.

Each Naive RAG chunk (~800 characters) is the parent; it is split into ~200-character children,
which are embedded on their own. Small children match a question sharply; the LLM then reads the
whole parents they came from. Search with a scalpel, read with a book.
"""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rags.naive_rag import NaiveRAG, Result, answer, load_store, save_store, with_similarity
from storage import Saved


class HierarchicalRAG:
    seconds_per_item = 0.02  # embedding ~5 children: measured with nomic-embed-text on Ollama

    def __init__(self, index: NaiveRAG, saved: Saved, k: int = 8, parents: int = 4):
        # k children are matched so that, after merging siblings, up to 4 distinct parents remain:
        # the same 4 x 800 characters of context Naive RAG gets, only matched more precisely.
        self.index, self.saved, self.k, self.parents = index, saved, k, parents
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        emb = index.store.embeddings
        self.children = load_store(saved, emb)

    def _todo(self) -> list[Document]:
        """Parents without children yet; drops children of parents a re-upload replaced."""
        parents = self.index.store.store
        orphans = [
            i for i, r in self.children.store.items() if r["metadata"]["parent"] not in parents
        ]
        if orphans:
            self.children.delete(orphans)
            save_store(self.children, self.saved)
        done = {r["metadata"]["parent"] for r in self.children.store.values()}
        return [c for c in self.index.chunks() if c.id not in done]

    def pending(self) -> int:
        return len(self._todo())

    def build(self, llm: BaseChatModel, progress: Callable[[int, int], None]) -> None:
        todo = self._todo()
        for i in range(0, len(todo), 50):  # save every 50 parents, so an interrupted build resumes
            batch = todo[i : i + 50]
            kids = [
                Document(text, metadata={**p.metadata, "parent": p.id})
                for p in batch
                for text in self.splitter.split_text(p.page_content)
            ]
            self.children.add_documents(kids)
            save_store(self.children, self.saved)
            progress(i + len(batch), len(todo))

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        hits = self.children.similarity_search(question, k=self.k)
        parent_ids = list(dict.fromkeys(h.metadata["parent"] for h in hits))[: self.parents]
        parents = self.index.store.get_by_ids(parent_ids)
        return answer(
            question, with_similarity(question, parents, self.index.store.embeddings), llm
        )
