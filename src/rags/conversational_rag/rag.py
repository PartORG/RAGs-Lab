"""Conversational RAG: condense the chat so far + the new turn into one standalone question.

Follow-ups ("and the second one?") are unsearchable on their own. With earlier turns, the LLM
first rewrites the new turn as a question that makes sense alone; that drives a normal search
and answer. The first turn of a chat is searched as is.
"""

from langchain_core.language_models import BaseChatModel

from rags.naive_rag import NaiveRAG, Result, answer

CONDENSE_PROMPT = """Conversation so far:
{transcript}

Rewrite the user's new message as one standalone question that makes sense without the
conversation. Reply with the question only.

New message: {question}"""


class ConversationalRAG:
    def __init__(self, index: NaiveRAG, history: list[tuple[str, str]], k: int = 4):
        # ponytail: only the last 5 turns are condensed; summarize older turns if long chats matter.
        self.index, self.history, self.k = index, history[-5:], k

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        standalone = question
        if self.history:
            transcript = "\n".join(f"user: {q}\nassistant: {a}" for q, a in self.history)
            prompt = CONDENSE_PROMPT.format(transcript=transcript, question=question)
            standalone = llm.invoke(prompt).text.strip()
        chunks = self.index.search(standalone, self.k)
        result = answer(standalone, chunks, llm)
        if self.history:
            result.trace = f"Standalone question: {standalone}"
        return result
