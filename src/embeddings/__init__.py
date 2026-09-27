"""Which model turns text into vectors — the choice underneath both chunking and retrieval.

An index belongs to the embedding that built it: the query has to be embedded by the same model,
in the same space, or the nearest neighbours mean nothing. So each embedding keeps its own
indexes, and switching asks to index the documents again.

`dimensions` is the vector width, which sets the index size; `ollama` marks the models Ollama
serves (no memory in this process) as opposed to the ones loaded here from Hugging Face.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

# Imported under private names so that, say, embeddings.nomic stays the folder, not the function.
from embeddings.bge_base.embedder import bge_base as _bge_base
from embeddings.bge_m3.embedder import bge_m3 as _bge_m3
from embeddings.bge_small.embedder import bge_small as _bge_small
from embeddings.minilm_multilingual.embedder import minilm_multilingual as _minilm_multilingual
from embeddings.mxbai.embedder import MODEL as MXBAI_MODEL
from embeddings.mxbai.embedder import mxbai as _mxbai
from embeddings.nomic.embedder import MODEL as NOMIC_MODEL
from embeddings.nomic.embedder import nomic as _nomic

__all__ = ["DEFAULT", "EMBEDDINGS", "Embedding"]


@dataclass(frozen=True)
class Embedding:
    label: str
    describe: str
    dimensions: int
    ollama: str  # the Ollama model to pull, or "" when the model is loaded from Hugging Face
    build: Callable[[], Embeddings]


# The key is part of the index file names, so renaming one starts a fresh index. "nomic" is
# spelled with no suffix, because it is what the existing indexes were built with.
EMBEDDINGS: dict[str, Embedding] = {
    "nomic": Embedding(
        "nomic-embed-text (Ollama)",
        "768 dimensions, English, served by Ollama so it costs this app no memory.",
        768,
        NOMIC_MODEL,
        _nomic,
    ),
    "mxbai": Embedding(
        "mxbai-embed-large (Ollama)",
        "1024 dimensions, stronger than nomic, 670 MB. Needs `ollama pull mxbai-embed-large`.",
        1024,
        MXBAI_MODEL,
        _mxbai,
    ),
    "bge_small": Embedding(
        "bge-small-en-v1.5",
        "384 dimensions, 130 MB, English. The fastest to index with, and the smallest index.",
        384,
        "",
        _bge_small,
    ),
    "bge_base": Embedding(
        "bge-base-en-v1.5",
        "768 dimensions, 440 MB, English. Same width as nomic, for a like-for-like comparison.",
        768,
        "",
        _bge_base,
    ),
    "bge_m3": Embedding(
        "bge-m3 (multilingual)",
        "1024 dimensions, 100+ languages, long inputs. Heavy: ~2.2 GB of RAM in this process.",
        1024,
        "",
        _bge_m3,
    ),
    "minilm_multilingual": Embedding(
        "MiniLM multilingual",
        "384 dimensions, 50+ languages, 470 MB. The light multilingual option.",
        384,
        "",
        _minilm_multilingual,
    ),
}

DEFAULT = "nomic"
