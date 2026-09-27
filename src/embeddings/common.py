"""What the embedding folders share: a local sentence-transformers wrapper.

langchain-huggingface would bring the same thing as a dependency; this is the dozen lines of it
this lab uses. Models run on the CPU, because Ollama's chat model already fills the GPU.
"""

from functools import cache

from langchain_core.embeddings import Embeddings


@cache
def _model(name: str):
    from sentence_transformers import SentenceTransformer  # loads torch: only when one is picked

    return SentenceTransformer(name, device="cpu")


class LocalEmbeddings(Embeddings):
    """A Hugging Face embedding model, held in this process rather than in Ollama.

    query_prefix is the instruction some models are trained to expect on the query side only;
    using the wrong one (or none) quietly costs retrieval accuracy.
    """

    def __init__(self, name: str, query_prefix: str = ""):
        self.name, self.query_prefix = name, query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = _model(self.name).encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self.query_prefix + text])[0]
