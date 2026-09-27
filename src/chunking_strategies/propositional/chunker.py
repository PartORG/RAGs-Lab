"""Propositional chunking: rewrite the text into standalone facts, one per chunk.

"It was released in 1971" becomes "Computer Space was released in 1971", so a chunk means the
same thing out of context as in it. Retrieval gets very sharp and very small units; anything the
rewrite drops is gone, and it costs one LLM call per page.
"""

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from chunking_strategies.common import Chunker, capped, piece

log = logging.getLogger(__name__)

FACTS_PROMPT = """Rewrite this text as a list of standalone facts.

Each fact must make sense on its own: replace pronouns with the name they refer to, and keep
names, numbers and titles. Do not invent anything that is not in the text.

{text}"""


class Propositions(BaseModel):
    facts: list[str]


def propositional(llm: BaseChatModel) -> Chunker:
    def chunk(pages: list[Document]) -> list[Document]:
        extract = llm.with_structured_output(Propositions)
        out = []
        for page in pages:
            if not page.page_content.strip():
                continue
            try:
                facts = extract.invoke(FACTS_PROMPT.format(text=page.page_content)).facts
            except Exception as e:  # model output is untrusted: keep the page rather than lose it
                log.warning("No propositions for a page, kept whole: %r", e)
                facts = []
            if facts:
                out += [piece(page, fact.strip()) for fact in facts if fact.strip()]
            else:
                out += capped(page, page.page_content)
        return out

    return chunk
