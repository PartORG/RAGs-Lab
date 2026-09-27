"""nomic-embed-text, served by Ollama: this lab's default.

768 dimensions, trained for retrieval, and small enough (274 MB) to sit beside a chat model on
the GPU. Ollama serves it, so it costs the Streamlit process no memory.
"""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

MODEL = "nomic-embed-text"


def nomic() -> Embeddings:
    return OllamaEmbeddings(model=MODEL)
