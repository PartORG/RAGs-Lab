"""Token-based chunking: size measured the way a model measures it.

Characters are a proxy; context windows and embedding limits are counted in tokens, and the ratio
moves with the language and the amount of code or punctuation. This counts real tokens with the
reranker's tokenizer, which the lab already downloads.
"""

from functools import cache

from langchain_core.documents import Document

from chunking_strategies.common import Chunker, piece, sentences_of

TOKENS = 200  # ~800 characters of English prose
TOKENIZER = "BAAI/bge-reranker-base"


@cache
def _tokenizer():
    from transformers import AutoTokenizer  # heavy: only imported when this chunker is used

    return AutoTokenizer.from_pretrained(TOKENIZER)


def token_based(tokens: int = TOKENS) -> Chunker:
    """Whole sentences packed until the next one would cross the token budget."""

    def chunk(pages: list[Document]) -> list[Document]:
        tokenizer = _tokenizer()
        out = []
        for page in pages:
            group, used = "", 0
            for sentence in sentences_of(page.page_content):
                size = len(tokenizer.tokenize(sentence))
                if group and used + size > tokens:
                    out.append(piece(page, group))
                    group, used = "", 0
                group = f"{group} {sentence}" if group else sentence
                used += size
            if group:
                out.append(piece(page, group))
        return out

    return chunk
