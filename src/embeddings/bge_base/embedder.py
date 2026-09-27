"""BAAI/bge-base-en-v1.5: 768 dimensions, 440 MB, English.

The middle of the BGE family, and a direct size-for-size comparison against nomic-embed-text.
"""

from langchain_core.embeddings import Embeddings

from embeddings.common import LocalEmbeddings

MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def bge_base() -> Embeddings:
    return LocalEmbeddings(MODEL, QUERY_PREFIX)
