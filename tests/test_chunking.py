import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from chunking_strategies import CHUNKERS, DEFAULT, MAX_CHUNK
from chunking_strategies.contextual_headers import contextual_headers
from chunking_strategies.document_structure import document_structure
from chunking_strategies.fixed_size import fixed_size
from chunking_strategies.llm_boundaries import llm_boundaries
from chunking_strategies.paragraphs import paragraphs
from chunking_strategies.propositional import propositional
from chunking_strategies.recursive import recursive
from chunking_strategies.semantic import semantic
from chunking_strategies.sentence_window import sentence_window
from chunking_strategies.sentences import sentences
from chunking_strategies.token_based import chunker as token_module
from chunking_strategies.token_based import token_based

PAGE = Document(
    " ".join(f"Sentence number {i} about pong." for i in range(60)),
    metadata={"source": "a.pdf", "page": 7},
)


def test_every_chunker_keeps_source_and_page_on_each_chunk():
    for key, chunking in CHUNKERS.items():
        if key == "tokens":
            continue  # would download a tokenizer; covered below with a stand-in
        chunks = chunking.build(FakeEmbeddings(), FakeLLM(responses=["2"]))([PAGE])
        assert chunks, key
        assert all(c.metadata["source"] == "a.pdf" and c.metadata["page"] == 7 for c in chunks), key


def test_fixed_cuts_every_n_characters_and_recursive_does_not():
    text = Document("word " * 400, metadata={"source": "a.txt"})
    blunt = fixed_size(size=100)([text])
    assert [len(c.page_content) for c in blunt[:-1]] == [100] * (len(blunt) - 1)
    assert all(len(c.page_content) <= 100 for c in recursive(size=100, overlap=0)([text]))


def test_sentences_never_splits_one_in_half():
    chunks = sentences(size=200)([PAGE])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page_content.endswith(".")
        assert chunk.page_content.startswith("Sentence")


def test_structure_keeps_a_markdown_section_together_and_caps_a_long_page():
    md = Document(
        "# Boing\nA remake of Pong.\n\n## Cavern\nA platformer.", metadata={"source": "a.md"}
    )
    sections = [c.page_content for c in document_structure()([md])]
    assert len(sections) == 2 and sections[0].startswith("# Boing")
    huge = Document("x" * (MAX_CHUNK * 2 + 10), metadata={"source": "a.pdf", "page": 1})
    assert all(len(c.page_content) <= MAX_CHUNK for c in document_structure()([huge]))


class FakeEmbeddings(Embeddings):
    """Sentences about the same word embed alike, so a change of subject is a real breakpoint."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        topics = ["pong", "cavern", "boing"]
        vector = np.zeros(4)
        for i, topic in enumerate(topics):
            vector[i] = 1.0 if topic in text.lower() else 0.0
        vector[3] = 0.1
        return list(vector)


def test_semantic_cuts_where_the_subject_changes():
    pong = " ".join(f"Pong sentence {i} is about pong and pong." for i in range(8))
    cavern = " ".join(f"Cavern sentence {i} is about cavern and cavern." for i in range(8))
    page = Document(f"{pong} {cavern}", metadata={"source": "a.txt"})
    chunks = semantic(FakeEmbeddings(), size=200)([page])
    assert len(chunks) >= 2
    mixed = [
        c for c in chunks if "pong" in c.page_content.lower() and "cavern" in c.page_content.lower()
    ]
    assert len(mixed) <= 1  # at most the one chunk that straddles the boundary


def test_the_default_chunking_is_the_one_the_indexes_were_built_with():
    assert DEFAULT == "recursive"
    chunks = CHUNKERS[DEFAULT].build(FakeEmbeddings(), FakeLLM(responses=["2"]))([PAGE])
    assert all(len(c.page_content) <= 800 for c in chunks)


class FakeLLM(FakeListChatModel):
    """Answers both kinds of chunking prompt: sentence numbers, and a list of facts."""

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda prompt: schema(facts=["Fact one.", "Fact two."]))


def test_sentence_window_stores_each_sentence_with_its_neighbours():
    page = Document("One. Two. Three. Four.", metadata={"source": "a.txt"})
    chunks = [c.page_content for c in sentence_window(window=1)([page])]
    assert chunks == ["One. Two.", "One. Two. Three.", "Two. Three. Four.", "Three. Four."]


def test_paragraphs_merge_up_to_the_size_and_never_split_one():
    page = Document("Alpha para.\n\nBeta para.\n\n" + "x" * 900, metadata={"source": "a.txt"})
    chunks = [c.page_content for c in paragraphs(size=100)([page])]
    assert chunks[0] == "Alpha para.\n\nBeta para."  # both small ones together
    assert chunks[1] == "x" * 900  # one paragraph, left whole even though it is over the size


def test_token_based_packs_sentences_to_a_token_budget(monkeypatch):
    class Whitespace:  # stands in for the real tokenizer
        def tokenize(self, text):
            return text.split()

    monkeypatch.setattr(token_module, "_tokenizer", lambda: Whitespace())
    page = Document("One two three. Four five six. Seven eight nine.", metadata={"source": "a.txt"})
    chunks = [c.page_content for c in token_based(tokens=6)([page])]
    assert chunks == ["One two three. Four five six.", "Seven eight nine."]


def test_llm_boundaries_splits_where_the_model_says():
    page = Document("One. Two. Three. Four. Five.", metadata={"source": "a.txt"})
    chunks = [c.page_content for c in llm_boundaries(FakeLLM(responses=["3"]))([page])]
    assert chunks == ["One. Two. Three.", "Four. Five."]


def test_propositional_turns_a_page_into_standalone_facts():
    page = Document("It was released in 1971 by them.", metadata={"source": "a.txt", "page": 2})
    chunks = propositional(FakeLLM(responses=["unused"]))([page])
    assert [c.page_content for c in chunks] == ["Fact one.", "Fact two."]
    assert all(c.metadata["page"] == 2 for c in chunks)


def test_contextual_headers_prefix_every_chunk():
    page = Document("Boing is a remake of Pong. " * 60, metadata={"source": "a.txt"})
    chunks = contextual_headers(FakeLLM(responses=["From the Boing chapter."]), size=200)([page])
    assert len(chunks) > 1
    assert all(c.page_content.startswith("From the Boing chapter.\n\n") for c in chunks)
