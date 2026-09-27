"""Iterative / Multi-hop RAG: investigate instead of look up (IRCoT-style).

For questions no single search answers, the LLM names the next sub-question, which is searched
and answered briefly; that note shapes the next sub-question. After at most 3 hops, or when the
LLM says the notes suffice, they are synthesized into the final answer.
"""

from langchain_core.language_models import BaseChatModel

from rags.naive_rag import NaiveRAG, Result, context_of, graded, with_similarity

NEXT_PROMPT = """Main question: {question}

Facts gathered so far:
{notes}

State the single next sub-question to look up in the documents, or reply DONE if the facts
already answer the main question. Reply with the sub-question or DONE only."""

HOP_PROMPT = """Context:
{context}

Using only the context, answer briefly: {question}
If the context does not say, reply "not found"."""

SYNTHESIS_PROMPT = """Evidence gathered from the documents:
{notes}

Using only this evidence, answer the question: {question}
If the evidence does not contain the answer, say "I don't know based on the documents." """


class MultiHopRAG:
    def __init__(self, index: NaiveRAG, max_hops: int = 3, k: int = 3):
        self.index, self.max_hops, self.k = index, max_hops, k

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        notes: list[str] = []
        seen = {}  # every chunk read on any hop, first-seen order
        for _ in range(self.max_hops):
            known = "\n\n".join(notes) or "(nothing yet)"
            sub = llm.invoke(NEXT_PROMPT.format(question=question, notes=known)).text.strip()
            if sub.upper().startswith("DONE"):
                if notes:
                    break
                sub = question  # "done" before looking anything up: look up the question itself
            hits = self.index.search(sub, self.k)
            prompt = HOP_PROMPT.format(context=context_of(hits), question=sub)
            note = llm.invoke(prompt).text.strip()
            notes.append(f"Q: {sub}\nA: {note}")
            for doc, _ in hits:
                seen.setdefault(doc.id, doc)
        evidence = "\n\n".join(notes)
        reply = llm.invoke(SYNTHESIS_PROMPT.format(notes=evidence, question=question)).text
        chunks = with_similarity(question, list(seen.values()), self.index.store.embeddings)
        result = graded(question, chunks, reply, llm)
        result.trace = evidence
        return result
