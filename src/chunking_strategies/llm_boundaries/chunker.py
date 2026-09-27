"""Agentic chunking: the model reads the page and says where the topics change.

Numbered sentences go to the LLM, which replies with the sentence numbers that start a new
topic. It is the most flexible way to split prose and the slowest: one LLM call per page.
"""

import logging
import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from chunking_strategies.common import MAX_CHUNK, Chunker, capped, piece, sentences_of

log = logging.getLogger(__name__)

SPLIT_PROMPT = """Here are the numbered sentences of one page:

{numbered}

Which sentences begin a new topic? Reply with their numbers only, comma-separated, in order, and
nothing else. Reply with 0 if the whole page is one topic."""


def llm_boundaries(llm: BaseChatModel) -> Chunker:
    def chunk(pages: list[Document]) -> list[Document]:
        out = []
        for page in pages:
            parts = sentences_of(page.page_content)
            if len(parts) < 3:
                out += capped(page, page.page_content)
                continue
            numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(parts))
            try:
                reply = llm.invoke(SPLIT_PROMPT.format(numbered=numbered)).text
            except Exception as e:  # a page the model chokes on stays one chunk
                log.warning("No boundaries for a page, kept as one chunk: %r", e)
                reply = ""
            starts = sorted({int(n) for n in re.findall(r"\d+", reply) if 0 < int(n) < len(parts)})
            group: list[str] = []
            for i, part in enumerate(parts):
                if i in starts and group:
                    out += capped(page, " ".join(group))
                    group = []
                group.append(part)
                if len(" ".join(group)) >= MAX_CHUNK:
                    out.append(piece(page, " ".join(group)))
                    group = []
            if group:
                out += capped(page, " ".join(group))
        return out

    return chunk
