"""Sentence chunking: pack whole sentences, never cutting one in half."""

from langchain_core.documents import Document

from chunking_strategies.common import SIZE, Chunker, piece, sentences_of


def sentences(size: int = SIZE) -> Chunker:
    def chunk(pages: list[Document]) -> list[Document]:
        out = []
        for page in pages:
            group = ""
            for sentence in sentences_of(page.page_content):
                if group and len(group) + len(sentence) + 1 > size:
                    out.append(piece(page, group))
                    group = ""
                group = f"{group} {sentence}" if group else sentence
            if group:
                out.append(piece(page, group))
        return out

    return chunk
