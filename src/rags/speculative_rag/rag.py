"""Speculative RAG: a small model drafts, the big one only verifies.

Each retrieved chunk goes to a small fast model, which drafts an answer from that chunk alone
plus the reason it fits; the big model then reads the drafts once and picks the best-supported
one. The big model writes a number instead of a whole answer, so it runs briefly.
"""

import re

from langchain_core.language_models import BaseChatModel

from rags.naive_rag import NaiveRAG, Result, graded

DRAFT_PROMPT = """Context:
{context}

Question: {question}
Answer in two or three sentences using only the context, then add one sentence naming what in the
context supports it. If the context does not answer the question, say so."""

PICK_PROMPT = """Question: {question}

Candidate answers:
{menu}

Reply with only the number of the best-supported answer."""


class SpeculativeRAG:
    def __init__(self, index: NaiveRAG, drafter: BaseChatModel, k: int = 3):
        self.index, self.drafter, self.k = index, drafter, k

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        chunks = self.index.search(question, self.k)
        # ponytail: drafted one after another; self.drafter.batch(...) if the wait matters.
        drafts = [
            self.drafter.invoke(
                DRAFT_PROMPT.format(context=doc.page_content, question=question)
            ).text.strip()
            for doc, _ in chunks
        ]
        if not drafts:
            return graded(question, chunks, "I don't know based on the documents.", llm)
        menu = "\n\n".join(f"[{i}] {draft}" for i, draft in enumerate(drafts))
        pick = llm.invoke(PICK_PROMPT.format(question=question, menu=menu)).text
        number = re.search(r"\d+", pick)
        chosen = int(number.group()) if number else 0
        chosen = chosen if chosen < len(drafts) else 0
        result = graded(question, chunks, drafts[chosen], llm)
        result.trace = f"The verifier picked draft [{chosen}] of {len(drafts)}.\n\n{menu}"
        return result
