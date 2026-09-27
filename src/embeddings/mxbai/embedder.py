"""mxbai-embed-large, served by Ollama: a larger, stronger alternative to nomic.

1024 dimensions and 670 MB. Pull it first: `ollama pull mxbai-embed-large`.
"""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

MODEL = "mxbai-embed-large"


def mxbai() -> Embeddings:
    return OllamaEmbeddings(model=MODEL)
