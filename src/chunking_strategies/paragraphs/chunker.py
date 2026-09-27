"""Paragraph chunking: the author's own unit of thought, a blank line apart."""

from langchain_core.documents import Document

from chunking_strategies.common import SIZE, Chunker, capped


def paragraphs(size: int = SIZE) -> Chunker:
    """Short neighbouring paragraphs are merged up to size; long ones are capped, never split
    at a character count if a paragraph break is available."""

    def chunk(pages: list[Document]) -> list[Document]:
        out = []
        for page in pages:
            group = ""
            for para in (p.strip() for p in page.page_content.split("\n\n") if p.strip()):
                if group and len(group) + len(para) + 2 > size:
                    out += capped(page, group)
                    group = ""
                group = f"{group}\n\n{para}" if group else para
            if group:
                out += capped(page, group)
        return out

    return chunk
