"""paraphrase-multilingual-MiniLM-L12-v2: 384 dimensions, 50+ languages, 470 MB.

For documents that are not in English when bge-m3 is too heavy for the machine.
"""

from langchain_core.embeddings import Embeddings

from embeddings.common import LocalEmbeddings

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def minilm_multilingual() -> Embeddings:
    return LocalEmbeddings(MODEL)
