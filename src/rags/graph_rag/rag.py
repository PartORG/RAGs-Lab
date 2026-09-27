"""GraphRAG: the LLM turns every chunk into (subject, relation, object) facts forming a graph; the
facts within 2 hops of the question's entities become the context.

Vector search finds passages that sound like the question; a graph answers "how is A connected
to B" by walking edges, and every fact it uses is traceable to the chunk it came from. This is
the local NetworkX version: no community summaries (Microsoft's global variant).
"""

import logging
from collections.abc import Callable

import networkx as nx
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from rags.naive_rag import NaiveRAG, Result, answer, with_similarity
from storage import Saved

log = logging.getLogger(__name__)

# No example triple in here: with one, qwen3 returned no facts at all for 2 in 3 chunks that
# plainly stated some (measured on 12 chunks of Code the Classics).
EXTRACT_PROMPT = """Extract the facts stated in this text as (subject, relation, object) triples.
Use short lowercase names for subjects and objects.

{text}"""

ENTITIES_PROMPT = """List the names and things this question is about: comma-separated, lowercase,
nothing else.

Question: {question}"""


class Facts(BaseModel):
    facts: list[tuple[str, str, str]]  # (subject, relation, object)


class GraphRAG:
    seconds_per_item = 3.5  # one structured LLM extraction per chunk, measured with qwen3:8b

    def __init__(self, index: NaiveRAG, saved: Saved, hops: int = 2, max_facts: int = 40):
        self.index, self.saved, self.hops, self.max_facts = index, saved, hops, max_facts
        state = saved.load() or {"done": [], "edges": []}
        self.done: set[str] = set(state["done"])  # chunk ids already extracted
        self.edges: list[list[str]] = state["edges"]  # [subject, relation, object, chunk id]

    def _save(self) -> None:
        self.saved.save({"done": sorted(self.done), "edges": self.edges})

    def _todo(self) -> list[Document]:
        """Chunks not extracted yet; drops facts of chunks a re-upload replaced."""
        chunks = self.index.store.store
        if not self.done <= chunks.keys():
            self.done &= chunks.keys()
            self.edges = [e for e in self.edges if e[3] in chunks]
            self._save()
        return [c for c in self.index.chunks() if c.id not in self.done]

    def pending(self) -> int:
        return len(self._todo())

    def build(self, llm: BaseChatModel, progress: Callable[[int, int], None]) -> None:
        todo = self._todo()
        extract = llm.with_structured_output(Facts)
        for i, chunk in enumerate(todo, 1):
            try:
                facts = extract.invoke(EXTRACT_PROMPT.format(text=chunk.page_content)).facts
            except Exception as e:  # model output is untrusted: that chunk adds no facts
                log.warning("No facts extracted from a chunk: %r", e)
                facts = []
            for subject, relation, obj in facts:
                subject, obj = subject.strip().lower(), obj.strip().lower()
                if subject and obj:
                    self.edges.append([subject, relation.strip(), obj, chunk.id])
            self.done.add(chunk.id)
            if i % 10 == 0 or i == len(todo):  # save every 10 chunks, so a build resumes
                self._save()
                progress(i, len(todo))

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        reply = llm.invoke(ENTITIES_PROMPT.format(question=question)).text
        entities = {e.strip().lower() for e in reply.split(",") if e.strip()}
        graph = nx.Graph([(s, o) for s, _, o, _ in self.edges])
        # An entity matches a node of the same name, or one containing it ("ball" -> "the ball").
        seeds = {n for n in graph if any(e == n or (len(e) > 3 and e in n) for e in entities)}
        hops = nx.multi_source_dijkstra_path_length(graph, seeds, cutoff=self.hops) if seeds else {}
        near = [e for e in self.edges if e[0] in hops and e[2] in hops]
        first_source: dict[tuple[str, str, str], str] = {}  # one row per fact, however often seen
        for s, r, o, c in near:
            first_source.setdefault((s, r, o), c)
        chunks = self.index.store.store
        facts = [
            Document(f"{s} -[{r}]-> {o}", metadata=dict(chunks[c]["metadata"]))
            for (s, r, o), c in first_source.items()
        ]
        # Two hops around a hub entity ("company") reach hundreds of facts, most of them noise,
        # so the ones that go to the LLM are those closest to the question, not the nearest hops.
        scored = with_similarity(question, facts, self.index.store.embeddings)
        scored.sort(key=lambda t: t[1], reverse=True)
        result = answer(question, scored[: self.max_facts], llm)
        result.trace = (
            f"Entities in the question: {', '.join(sorted(entities))}\n"
            f"Matching graph nodes: {len(seeds)}; facts within {self.hops} hops: {len(near)}"
        )
        return result
