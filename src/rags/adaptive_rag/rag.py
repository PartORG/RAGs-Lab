"""Adaptive RAG: a router sends each question down the cheapest path that can answer it.

One classification call picks: NONE (small talk, general knowledge) -> the model answers with no
retrieval; SIMPLE (one lookup) -> Naive RAG; COMPLEX (several facts combined) -> Iterative /
Multi-hop RAG. Misrouting a hard question to the cheap path is the price of the savings.
"""

from langchain_core.language_models import BaseChatModel

from rags.multi_hop_rag import MultiHopRAG
from rags.naive_rag import NaiveRAG, Result, graded

# NONE is deliberately narrow: asked to judge whether it "needs" the documents, qwen3 routed
# questions about the documents to itself and answered them from memory.
ROUTE_PROMPT = """The user is asking about documents they uploaded. Classify their message.

NONE = small talk or a request that clearly concerns no document (a greeting, thanks, arithmetic)
SIMPLE = one lookup in the documents answers it
COMPLEX = needs several facts from the documents combined

Reply with one word: NONE, SIMPLE or COMPLEX.

Message: {question}"""


class AdaptiveRAG:
    def __init__(self, index: NaiveRAG):
        self.index = index

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        label = llm.invoke(ROUTE_PROMPT.format(question=question)).text.upper()
        if "NONE" in label:
            result = graded(question, [], llm.invoke(question).text, llm)
            route = "NONE: answered directly, no retrieval"
        elif "SIMPLE" in label:
            result = self.index.ask(question, llm)
            route = "SIMPLE: Naive RAG"
        else:
            result = MultiHopRAG(self.index).ask(question, llm)
            route = "COMPLEX: Iterative / Multi-hop RAG"
        result.trace = f"Router: {route}" + (f"\n\n{result.trace}" if result.trace else "")
        return result
