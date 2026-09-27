"""Agentic RAG: searching is a tool the model may call, not a fixed step.

The model decides whether to search at all, with what wording, and whether to search again after
seeing the results — up to a step cap, because an agent with no cap can loop. A hand-rolled tool
loop (bind_tools plus one tool) does this in a few lines; no agent framework needed.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from rags.naive_rag import NaiveRAG, Result, context_of, graded

SYSTEM = """You answer questions about the user's own documents.

Use the search_documents tool to look things up, and call it again with different wording if the
passages are not enough. Answer using only what the documents said; if they do not contain the
answer, say "I don't know based on the documents."."""

LAST_CALL = "Answer now, using only what the documents said."


class AgenticRAG:
    def __init__(self, index: NaiveRAG, k: int = 4, max_steps: int = 4):
        self.index, self.k, self.max_steps = index, k, max_steps

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        found: dict[str, tuple] = {}  # every chunk any search returned, first-seen order
        searches: list[str] = []

        @tool
        def search_documents(query: str) -> str:
            """Search the user's documents and return the most relevant passages."""
            searches.append(query)
            hits = self.index.search(query, self.k)
            for doc, score in hits:
                found.setdefault(doc.id, (doc, score))
            return context_of(hits) or "(nothing found)"

        agent = llm.bind_tools([search_documents])
        messages = [SystemMessage(SYSTEM), HumanMessage(question)]
        reply = AIMessage("")
        for _ in range(self.max_steps):
            reply = agent.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                passages = search_documents.invoke(call["args"])
                messages.append(ToolMessage(passages, tool_call_id=call["id"]))
        else:  # out of steps while still calling tools: make it answer with what it has
            reply = llm.invoke([*messages, HumanMessage(LAST_CALL)])
        result = graded(question, list(found.values()), reply.text, llm)
        result.trace = (
            "Searches the model chose:\n" + "\n".join(f"- {s}" for s in searches)
            if searches
            else "The model answered without searching."
        )
        return result
