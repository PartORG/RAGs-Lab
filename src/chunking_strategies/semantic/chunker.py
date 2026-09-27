"""Semantic chunking: cut where the text changes subject, not where a counter runs out.

Every sentence is embedded; the boundaries whose neighbours are least alike become the cuts. The
cost is one embedding per sentence at indexing time.
"""

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from chunking_strategies.common import MAX_CHUNK, SIZE, Chunker, capped, piece, sentences_of


def semantic(embeddings: Embeddings, size: int = SIZE, breakpoint: float = 0.3) -> Chunker:
    """breakpoint is the share of boundaries treated as subject changes: 0.3 = the lowest 30%."""

    def chunk(pages: list[Document]) -> list[Document]:
        out = []
        for page in pages:
            parts = sentences_of(page.page_content)
            if len(parts) < 3:
                out += capped(page, page.page_content)
                continue
            vectors = np.array(embeddings.embed_documents(parts))
            units = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            similarity = (units[:-1] * units[1:]).sum(axis=1)
            cut_below = np.quantile(similarity, breakpoint)
            group: list[str] = []
            for i, part in enumerate(parts):
                group.append(part)
                text = " ".join(group)
                subject_changed = i < len(similarity) and similarity[i] <= cut_below
                if len(text) >= MAX_CHUNK or (subject_changed and len(text) >= size / 2):
                    out.append(piece(page, text))
                    group = []
            if group:
                out.append(piece(page, " ".join(group)))
        return out

    return chunk
