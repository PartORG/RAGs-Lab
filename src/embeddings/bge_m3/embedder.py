"""BAAI/bge-m3: 1024 dimensions, multilingual, 8192-token inputs.

The strongest of the local options here and the heaviest: ~2.2 GB of RAM in this process, next to
whatever Ollama is holding. It needs no query prefix.
"""

from langchain_core.embeddings import Embeddings

from embeddings.common import LocalEmbeddings

MODEL = "BAAI/bge-m3"


def bge_m3() -> Embeddings:
    return LocalEmbeddings(MODEL)
