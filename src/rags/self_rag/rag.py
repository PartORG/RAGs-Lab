"""Self-RAG: the model critiques its own pipeline at every step.

Does this question need the documents at all? Are the retrieved chunks relevant (if not, rewrite
the query and search again)? Is the draft answer grounded in them (if not, answer again, told
what went wrong)? Each critique is one YES/NO call; the loops are capped so they cannot spin.
The paper fine-tunes reflection tokens into the model; this is the prompt-based version.
"""

from langchain_core.language_models import BaseChatModel

from rags.corrective_rag.rag import REWRITE_PROMPT, is_yes
from rags.naive_rag import ANSWER_PROMPT, NaiveRAG, Result, context_of, graded

# Asked the other way round ("does this need the documents?"), qwen3 claimed it already knew the
# answer to questions about the documents, and answered from memory. This asks it to justify
# skipping them instead.
NEEDS_DOCS_PROMPT = """The user is asking about documents they uploaded.

Is this message small talk or a general request that clearly has nothing to do with any document
(a greeting, thanks, arithmetic)? Answer YES or NO only.

Message: {question}"""

RELEVANT_PROMPT = """Question: {question}

Chunks:
{context}

Do these chunks contain information that helps answer the question? Answer YES or NO only."""

GROUNDED_PROMPT = """Context:
{context}

Answer: {answer}

Is every claim in the answer supported by the context? Answer YES or NO only."""

RETRY_PROMPT = (
    ANSWER_PROMPT
    + """

Your previous answer made claims the context does not support:
{previous}
Answer again, stating only what the context says."""
)


class SelfRAG:
    def __init__(self, index: NaiveRAG, k: int = 4, max_rewrites: int = 1, max_drafts: int = 2):
        self.index, self.k = index, k
        self.max_rewrites, self.max_drafts = max_rewrites, max_drafts

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        if is_yes(llm, NEEDS_DOCS_PROMPT.format(question=question)):
            result = graded(question, [], llm.invoke(question).text, llm)
            result.trace = "Small talk: answered without retrieval."
            return result

        trace = ["Needs the documents? YES"]
        query = question
        for attempt in range(self.max_rewrites + 1):
            chunks = self.index.search(query, self.k)
            prompt = RELEVANT_PROMPT.format(question=question, context=context_of(chunks))
            relevant = is_yes(llm, prompt)
            trace.append(f"Searched for: {query}\nChunks relevant? {'YES' if relevant else 'NO'}")
            if relevant or attempt == self.max_rewrites:
                break
            query = llm.invoke(REWRITE_PROMPT.format(question=question)).text.strip()

        context = context_of(chunks)
        reply = llm.invoke(ANSWER_PROMPT.format(context=context, question=question)).text
        for draft in range(1, self.max_drafts + 1):
            grounded = is_yes(llm, GROUNDED_PROMPT.format(context=context, answer=reply))
            trace.append(f"Draft {draft} grounded? {'YES' if grounded else 'NO'}")
            if grounded or draft == self.max_drafts:
                break
            prompt = RETRY_PROMPT.format(context=context, question=question, previous=reply)
            reply = llm.invoke(prompt).text
        result = graded(question, chunks, reply, llm)
        result.trace = "\n".join(trace)
        return result
