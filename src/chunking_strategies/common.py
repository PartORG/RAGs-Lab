"""What every chunker shares: the type it implements, the sizes, and two small helpers."""

import re
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

Chunker = Callable[[list[Document]], list[Document]]  # pages (or whole files) in, chunks out

SIZE = 800  # characters: the size the retrieval strategies were tuned against
OVERLAP = 100
MAX_CHUNK = 3000  # nomic-embed-text reads ~2048 tokens; a longer chunk would be cut off silently
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def piece(page: Document, text: str) -> Document:
    """A chunk carrying its page's metadata, so every strategy can still cite source and page."""
    return Document(text, metadata=dict(page.metadata))


def capped(page: Document, text: str) -> list[Document]:
    """Cut anything the embedding model would silently truncate."""
    if len(text) <= MAX_CHUNK:
        return [piece(page, text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK, chunk_overlap=0)
    return [piece(page, part) for part in splitter.split_text(text)]


def sentences_of(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text) if s.strip()]
