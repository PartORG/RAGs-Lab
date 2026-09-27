"""RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).

The Naive RAG chunks are clustered by embedding (Gaussian Mixture) and the LLM summarizes each
cluster; the summaries are clustered and summarized again. Chunks and summaries of every level
are searched together ("collapsed tree"), so a question can match one detail or a theme that no
single chunk states. The tree covers the whole corpus: any upload makes it stale.
"""

import hashlib
from collections.abc import Callable

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from sklearn.mixture import GaussianMixture

from rags.naive_rag import NaiveRAG, Result, answer
from storage import Saved

SUMMARY_PROMPT = """Summarize the key facts of these passages in one paragraph. Keep names,
titles and numbers.

{passages}"""


class RaptorRAG:
    seconds_per_item = 0.9  # ~1 summary per 4 chunks, ~4 s each: measured with qwen3:8b

    def __init__(self, index: NaiveRAG, saved: Saved, levels: int = 2, k: int = 4):
        self.index, self.saved, self.levels, self.k = index, saved, levels, k
        state = saved.load() or {"covers": [], "nodes": {}}
        self.covers = set(state["covers"])  # the chunk ids of the last complete tree
        self.summaries = InMemoryVectorStore(index.store.embeddings)
        self.summaries.store = state["nodes"]

    def _save(self) -> None:
        nodes = self.summaries.store
        self.saved.save({"covers": sorted(self.covers), "nodes": nodes})

    def pending(self) -> int:
        chunks = self.index.store.store
        return 0 if self.covers == chunks.keys() else len(chunks)

    def build(self, llm: BaseChatModel, progress: Callable[[int, int], None]) -> None:
        # ponytail: the whole tree is rebuilt on any change (summaries of unchanged clusters are
        # reused); re-cluster only the touched branch if the corpus churns.
        emb = self.index.store.embeddings
        chunks = self.index.store.store
        old, new = dict(self.summaries.store), {}
        layer = [(i, r["text"], r["vector"]) for i, r in chunks.items()]
        expected = max(2, len(layer) // 5) + max(2, len(layer) // 25)
        done = 0
        for level in range(1, self.levels + 1):
            if len(layer) < 4:
                break
            n = max(2, len(layer) // 5)
            # Diagonal covariance: a full 768x768 one per cluster is degenerate with ~5 points each.
            gmm = GaussianMixture(n, covariance_type="diag", random_state=0)
            labels = gmm.fit_predict(np.array([vector for _, _, vector in layer]))
            next_layer = []
            for cluster in range(n):
                members = [
                    node for node, label in zip(layer, labels, strict=True) if label == cluster
                ]
                if not members:  # GMM can leave a cluster empty
                    continue
                member_ids = sorted(i for i, _, _ in members)
                node_id = hashlib.sha1("|".join(member_ids).encode()).hexdigest()[:16]
                if node_id not in old:  # same members as a saved summary: an interrupted build
                    passages = "\n---\n".join(text for _, text, _ in members)
                    text = llm.invoke(SUMMARY_PROMPT.format(passages=passages)).text
                    old[node_id] = {
                        "id": node_id,
                        "vector": emb.embed_documents([text])[0],
                        "text": text,
                        "metadata": {"source": f"RAPTOR summary, level {level}"},
                    }
                    self.summaries.store = old
                    self._save()
                new[node_id] = old[node_id]
                next_layer.append((node_id, new[node_id]["text"], new[node_id]["vector"]))
                done += 1
                progress(min(done, expected), expected)
            layer = next_layer
        self.summaries.store, self.covers = new, set(chunks)  # drop summaries of the old tree
        self._save()

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        leaves = self.index.search(question, self.k)
        summaries = self.summaries.similarity_search_with_score(question, k=self.k)
        best = sorted(leaves + summaries, key=lambda t: t[1], reverse=True)[: self.k]
        return answer(question, best, llm)
