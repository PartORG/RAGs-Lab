"""HyDE (Hypothetical Document Embeddings): search with a fake ideal answer, not the question.

Questions and the passages that answer them are worded differently and sit in different regions
of embedding space. The LLM first drafts the passage it would expect to find; that draft lands
near the real answer passages. The draft itself is never shown to the answering step.
"""

from langchain_core.language_models import BaseChatModel

from rags.naive_rag import NaiveRAG, Result, answer

HYDE_PROMPT = """Write one short paragraph, in the style of the documentation or book it would
come from, that would perfectly answer this question: {question}"""


class HydeRAG:
    def __init__(self, index: NaiveRAG, k: int = 4):
        self.index, self.k = index, k

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        draft = llm.invoke(HYDE_PROMPT.format(question=question)).text
        # The draft is what gets searched; the retriever decides how that search happens.
        chunks = self.index.search(draft, self.k)
        result = answer(question, chunks, llm)
        result.trace = f"Hypothetical passage used for the search:\n{draft}"
        return result
