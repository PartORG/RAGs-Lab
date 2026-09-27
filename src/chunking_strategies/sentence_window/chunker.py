"""Sentence-window chunking: match on one sentence, read it with its neighbours.

Each chunk is one sentence plus the ones either side of it, so the embedding stays about a single
thought while the text handed to the model still has its context. Chunks overlap heavily by
design, which is the cost.
"""

from langchain_core.documents import Document

from chunking_strategies.common import Chunker, piece, sentences_of


def sentence_window(window: int = 2) -> Chunker:
    def chunk(pages: list[Document]) -> list[Document]:
        out = []
        for page in pages:
            parts = sentences_of(page.page_content)
            for i in range(len(parts)):
                start, end = max(0, i - window), min(len(parts), i + window + 1)
                out.append(piece(page, " ".join(parts[start:end])))
        return out

    return chunk
