"""Structure-aware chunking: follow the document's own divisions.

A Markdown section or a PDF page is a unit its author created, so its text belongs together. The
headings stay in the chunk, which also gives the embedding something to key on.
"""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from chunking_strategies.common import Chunker, capped

HEADINGS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def document_structure() -> Chunker:
    markdown = MarkdownHeaderTextSplitter(HEADINGS, strip_headers=False)

    def chunk(pages: list[Document]) -> list[Document]:
        out = []
        for page in pages:
            if page.metadata.get("page"):  # a PDF page is already a unit of layout
                out += capped(page, page.page_content)
                continue
            for section in markdown.split_text(page.page_content):
                out += capped(page, section.page_content)
        return out

    return chunk
