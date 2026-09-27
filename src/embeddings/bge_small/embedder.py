"""BAAI/bge-small-en-v1.5: 384 dimensions, 130 MB, English.

The cheap end of the quality curve — a third of the index size and the fastest to embed with,
which makes it the one to reach for when trying a chunking on a big corpus.
"""

from langchain_core.embeddings import Embeddings

from embeddings.common import LocalEmbeddings

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def bge_small() -> Embeddings:
    return LocalEmbeddings(MODEL, QUERY_PREFIX)
