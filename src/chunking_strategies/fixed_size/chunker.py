"""Fixed-size chunking: cut every N characters, wherever that lands."""

from langchain_core.documents import Document

from chunking_strategies.common import SIZE, Chunker, piece


def fixed_size(size: int = SIZE, overlap: int = 0) -> Chunker:
    def chunk(pages: list[Document]) -> list[Document]:
        step = size - overlap
        return [
            piece(page, page.page_content[start : start + size])
            for page in pages
            for start in range(0, max(len(page.page_content), 1), step)
        ]

    return chunk
