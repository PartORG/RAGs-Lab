import embeddings.common as common
from embeddings import DEFAULT, EMBEDDINGS
from embeddings.bge_small.embedder import QUERY_PREFIX
from embeddings.common import LocalEmbeddings


class StubModel:
    """Stands in for a sentence-transformers model: records what it was asked to encode."""

    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        import numpy as np

        self.seen += list(texts)
        return np.array([[float(len(t)), 1.0] for t in texts])


def test_the_query_prefix_goes_on_the_query_only(monkeypatch):
    stub = StubModel()
    monkeypatch.setattr(common, "_model", lambda name: stub)
    local = LocalEmbeddings("any/model", QUERY_PREFIX)

    local.embed_documents(["a chunk of text"])
    local.embed_query("a question")

    assert stub.seen == ["a chunk of text", QUERY_PREFIX + "a question"]


def test_embeddings_return_plain_lists_of_floats(monkeypatch):
    monkeypatch.setattr(common, "_model", lambda name: StubModel())
    [vector] = LocalEmbeddings("any/model").embed_documents(["abc"])
    assert vector == [3.0, 1.0]


def test_the_registry_is_consistent():
    assert DEFAULT == "nomic"  # what the existing indexes were built with
    labels = [e.label for e in EMBEDDINGS.values()]
    assert len(set(labels)) == len(labels)
    assert all(e.dimensions in (384, 768, 1024) for e in EMBEDDINGS.values())
    assert all(callable(e.build) for e in EMBEDDINGS.values())
    # Only the Ollama-served ones name a model to pull; the rest come from Hugging Face.
    assert {key for key, e in EMBEDDINGS.items() if e.ollama} == {"nomic", "mxbai"}
