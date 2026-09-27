import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import storage
from rags.naive_rag import NaiveRAG
from storage import Saved


@pytest.fixture(autouse=True)
def database(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite file."""
    monkeypatch.setattr(storage, "DATA", tmp_path)
    monkeypatch.setattr(storage, "DB", tmp_path / "lab.db")


@pytest.fixture
def make_index(tmp_path):
    """A Naive RAG index holding each text as one file of one chunk, on fake embeddings (the same
    text always embeds the same, so searching a chunk's exact text finds it first)."""

    def make(*texts: str) -> NaiveRAG:
        index = NaiveRAG(DeterministicFakeEmbedding(size=16), Saved("tester", "naive_rag"))
        for i, text in enumerate(texts):
            (tmp_path / f"{i}.txt").write_text(text)
            index.add_file(tmp_path / f"{i}.txt")
        return index

    return make
