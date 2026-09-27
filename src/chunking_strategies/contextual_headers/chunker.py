"""Contextual chunking: every chunk gets an LLM-written line saying where it sits.

Anthropic's contextual retrieval, applied at chunking time so that every strategy inherits it
rather than only Contextual RAG. Chunks are cut recursively first, then each is prefixed with one
sentence placing it in its document: one LLM call per chunk, the slowest chunker here.
"""

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from chunking_strategies.common import SIZE, Chunker, piece
from chunking_strategies.recursive import recursive

log = logging.getLogger(__name__)

HEADER_PROMPT = """<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

In one sentence, say where this chunk sits in the document and what it covers, to improve search
retrieval. Reply with that sentence only."""

DOCUMENT_CHARS = 8000  # the opening of the document: title, contents, introduction


def contextual_headers(llm: BaseChatModel, size: int = SIZE) -> Chunker:
    cut = recursive(size=size)

    def chunk(pages: list[Document]) -> list[Document]:
        start = "\n".join(page.page_content for page in pages)[:DOCUMENT_CHARS]
        out = []
        for piece_ in cut(pages):
            try:
                prompt = HEADER_PROMPT.format(document=start, chunk=piece_.page_content)
                header = llm.invoke(prompt).text.strip()
            except Exception as e:  # without its header the chunk is still usable
                log.warning("No header for a chunk, kept without one: %r", e)
                header = ""
            text = f"{header}\n\n{piece_.page_content}" if header else piece_.page_content
            out.append(piece(piece_, text))
        return out

    return chunk
