"""Recursive chunking: the same fixed size, but cut at the most natural boundary that fits."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking_strategies.common import OVERLAP, SIZE, Chunker


def recursive(size: int = SIZE, overlap: int = OVERLAP) -> Chunker:
    """Tries paragraph, then line, then sentence, then word boundaries, in that order."""
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents
