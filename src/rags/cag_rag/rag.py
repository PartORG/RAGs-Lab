"""Cache-Augmented Generation: no retrieval at all — the whole corpus goes in the prompt.

With every document in front of it, the model cannot retrieve the wrong passage; Ollama keeps the
KV cache of that identical prefix warm, so later questions skip re-reading it. The catch is the
context window: this refuses when the corpus does not fit, rather than silently losing the end of
it, which on a small local GPU is the binding constraint.
"""

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rags.naive_rag import NaiveRAG, Result, graded

SYSTEM = """Answer strictly from this knowledge base:

{corpus}"""


class CAGRag:
    def __init__(self, index: NaiveRAG, num_ctx: int = 16384, keep_alive: str = "30m"):
        self.index, self.num_ctx, self.keep_alive = index, num_ctx, keep_alive

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        chunks = self.index.chunks()
        # Each chunk keeps its file name and page: without them the model cannot answer "what does
        # page 3 say?", which is exactly the kind of question a whole-corpus prompt should nail.
        corpus = "\n\n".join(
            f"[{c.metadata['source']}"
            + (f", page {c.metadata['page']}]" if c.metadata.get("page") else "]")
            + f"\n{c.page_content}"
            for c in chunks
        )
        budget = (self.num_ctx - 1500) * 4  # ~4 characters per token, minus room for the answer
        if len(corpus) > budget:
            too_big = (
                f"The documents are about {len(corpus) // 4:,} tokens; CAG can only answer when "
                f"they fit the context window ({self.num_ctx:,} tokens here). Index fewer "
                "documents, or raise num_ctx if the GPU has the memory."
            )
            result = Result(too_big, [], None, "")
            result.trace = "Nothing was sent to the model."
            return result
        big = llm.model_copy(update={"num_ctx": self.num_ctx, "keep_alive": self.keep_alive})
        messages = [SystemMessage(SYSTEM.format(corpus=corpus)), HumanMessage(question)]
        reply = big.invoke(messages).text
        result = graded(question, self._scored(question, chunks), reply, big)
        result.trace = (
            f"No retrieval: all {len(chunks)} chunks (~{len(corpus) // 4:,} tokens) were in the "
            f"prompt, and Ollama keeps them cached for {self.keep_alive}."
        )
        return result

    def _scored(self, question: str, chunks: list) -> list[tuple]:
        """Chunks with their similarity to the question, from the vectors already in the index:
        the model saw all of them, so this only says which were worth its attention."""
        vectors = np.array([r["vector"] for r in self.index.store.store.values()])
        q = np.array(self.index.store.embeddings.embed_query(question))
        sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
        return list(zip(chunks, sims.tolist(), strict=True))
