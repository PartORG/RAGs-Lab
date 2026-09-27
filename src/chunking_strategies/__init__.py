"""How a document becomes chunks — the choice every retrieval strategy inherits.

Retrieval can only ever return a chunk, so where the cuts fall decides what an answer can be
built from. Each strategy lives in its own folder here; CHUNKING_STRATEGIES.md explains what each
one is for. Every chunking keeps its own indexes, so switching between them costs only the
re-indexing, and the same question can be asked of the same documents cut several ways.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Imported under private names so that, say, chunking_strategies.semantic stays the folder
# rather than the function inside it.
from chunking_strategies.common import MAX_CHUNK, OVERLAP, SIZE, Chunker
from chunking_strategies.contextual_headers.chunker import contextual_headers as _contextual_headers
from chunking_strategies.document_structure.chunker import document_structure as _document_structure
from chunking_strategies.fixed_size.chunker import fixed_size as _fixed_size
from chunking_strategies.llm_boundaries.chunker import llm_boundaries as _llm_boundaries
from chunking_strategies.paragraphs.chunker import paragraphs as _paragraphs
from chunking_strategies.propositional.chunker import propositional as _propositional
from chunking_strategies.recursive.chunker import recursive as _recursive
from chunking_strategies.semantic.chunker import semantic as _semantic
from chunking_strategies.sentence_window.chunker import sentence_window as _sentence_window
from chunking_strategies.sentences.chunker import sentences as _sentences
from chunking_strategies.token_based.chunker import token_based as _token_based

__all__ = ["CHUNKERS", "DEFAULT", "MAX_CHUNK", "OVERLAP", "SIZE", "Chunker", "Chunking"]


@dataclass(frozen=True)
class Chunking:
    label: str
    describe: str
    seconds_per_page: float  # indexing cost per page, measured on this laptop with qwen3:8b
    build: Callable[[Embeddings, BaseChatModel], Chunker]


# The key is part of the index file names, so renaming one starts a fresh index. "recursive" is
# spelled with no suffix, because it is what the existing indexes were built with.
CHUNKERS: dict[str, Chunking] = {
    "recursive": Chunking(
        "Recursive characters",
        f"{SIZE} characters, {OVERLAP} overlap, cut at the nearest paragraph or sentence.",
        0.01,
        lambda embeddings, llm: _recursive(),
    ),
    "fixed": Chunking(
        "Fixed characters",
        f"Every {SIZE} characters exactly, mid-sentence if that is where it lands.",
        0.01,
        lambda embeddings, llm: _fixed_size(),
    ),
    "sentences": Chunking(
        "Whole sentences",
        f"Sentences packed up to {SIZE} characters; none is ever cut in half.",
        0.01,
        lambda embeddings, llm: _sentences(),
    ),
    "window": Chunking(
        "Sentence window",
        "One chunk per sentence, stored with the two sentences either side of it.",
        0.01,
        lambda embeddings, llm: _sentence_window(),
    ),
    "paragraphs": Chunking(
        "Paragraphs",
        f"The author's own paragraphs, merged up to {SIZE} characters.",
        0.01,
        lambda embeddings, llm: _paragraphs(),
    ),
    "structure": Chunking(
        "Document structure",
        "One chunk per Markdown section or PDF page, split only if very long.",
        0.01,
        lambda embeddings, llm: _document_structure(),
    ),
    "tokens": Chunking(
        "Token budget",
        "Whole sentences packed to 200 real tokens, counted with a tokenizer.",
        0.05,
        lambda embeddings, llm: _token_based(),
    ),
    "semantic": Chunking(
        "Semantic breakpoints",
        "Cuts where the subject changes, by embedding every sentence.",
        0.35,
        lambda embeddings, llm: _semantic(embeddings),
    ),
    "llm": Chunking(
        "Model-chosen boundaries",
        "The model reads each page and says where the topics change. Slow.",
        0.8,
        lambda embeddings, llm: _llm_boundaries(llm),
    ),
    "propositions": Chunking(
        "Propositions",
        "The model rewrites each page as standalone facts, one per chunk. Slow.",
        11.0,
        lambda embeddings, llm: _propositional(llm),
    ),
    "contextual": Chunking(
        "Contextual headers",
        "Recursive chunks, each prefixed with a line placing it in the document. Slowest.",
        3.5,
        lambda embeddings, llm: _contextual_headers(llm),
    ),
}

DEFAULT = "recursive"
