"""Self-Query RAG: the question's hard constraints become filters, not similarity.

"What does page 40 of the book say about the ball?" mixes a semantic part with an exact one, and
cosine similarity cannot enforce the exact one. An LLM splits the question into search text plus
filters over the metadata this lab keeps — file name and page number — and only chunks that pass
the filters are searched at all.
"""

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from rags.naive_rag import NaiveRAG, Result, answer, with_similarity

log = logging.getLogger(__name__)

# The file names go in the prompt, and a name that matches no file is dropped below: asked
# without them, qwen3 invented sources like "user's documents" and filtered everything away.
PARSE_PROMPT = """These files are indexed, each chunk knowing its file name and, for PDFs, its
page number:
{sources}

Split this question into the text to search for and any hard filters it states. Use a file name
filter only if the question names one of the files above, and a page filter only if it states a
page. Leave a filter out otherwise.

Question: {question}"""


class Search(BaseModel):
    text: str  # what to search for semantically
    source: str | None = None  # part of a file name, when the question names a document
    page_min: int | None = None
    page_max: int | None = None


class SelfQueryRAG:
    def __init__(self, index: NaiveRAG, k: int = 4):
        self.index, self.k = index, k

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        names = sorted(self.index.sources())
        try:
            prompt = PARSE_PROMPT.format(
                sources="\n".join(f"- {n}" for n in names), question=question
            )
            search = llm.with_structured_output(Search).invoke(prompt)
        except Exception as e:  # model output is untrusted: fall back to a plain search
            log.warning("Self-Query could not parse the question, searching plainly: %r", e)
            search = Search(text=question)
        invented = ""
        if search.source and not any(search.source.lower() in n.lower() for n in names):
            invented = search.source  # a file name that does not exist would filter out everything
            search.source = None

        def keep(doc: Document) -> bool:
            page = doc.metadata.get("page")
            source = doc.metadata.get("source", "")
            if search.source and search.source.lower() not in source.lower():
                return False
            if search.page_min and (page is None or page < search.page_min):
                return False
            return not (search.page_max and (page is None or page > search.page_max))

        filters = {
            name: value
            for name, value in [
                ("file name contains", search.source),
                ("page from", search.page_min),
                ("page to", search.page_max),
            ]
            if value
        }
        docs = self.index.store.similarity_search(search.text or question, k=self.k, filter=keep)
        chunks = with_similarity(question, docs, self.index.store.embeddings)
        result = answer(question, chunks, llm)
        result.trace = f"Searched for: {search.text!r}\nFilters: " + (
            ", ".join(f"{name} {value}" for name, value in filters.items()) if filters else "none"
        )
        if invented:
            result.trace += f"\nIgnored a filter for {invented!r}: no such file is indexed."
        if filters and not docs:
            result.trace += "\nNo chunk passed the filters."
        return result
