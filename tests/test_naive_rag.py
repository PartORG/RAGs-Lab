import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from rags.naive_rag import NaiveRAG, remove_everywhere
from storage import Saved


def test_reindex_persist_and_ask(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("The retention period is 30 days.")
    index = Saved("tester", "naive_rag")
    emb = DeterministicFakeEmbedding(size=16)

    rag = NaiveRAG(emb, index)
    assert rag.add_file(doc) == 1
    assert rag.add_file(doc) == 1  # re-upload replaces the old chunks instead of duplicating
    assert NaiveRAG(emb, index).sources() == {"notes.txt": 1}  # survives a restart

    result = rag.ask("The retention period is 30 days.", FakeListChatModel(responses=["30 days"]))
    assert result.answer == "30 days"
    assert result.chunks[0][1] == pytest.approx(1.0)  # identical text -> cosine similarity 1
    assert result.grade is None and result.grade_error  # a failed grade is reported, not raised


def test_removing_a_file_drops_its_chunks_and_leaves_the_others(tmp_path):
    keep, drop = tmp_path / "keep.txt", tmp_path / "drop.txt"
    keep.write_text("The retention period is 30 days.")
    drop.write_text("Nothing to see here.")
    index = Saved("tester", "naive_rag")
    emb = DeterministicFakeEmbedding(size=16)

    rag = NaiveRAG(emb, index)
    rag.add_file(keep)
    rag.add_file(drop)
    assert rag.remove_file("drop.txt") == 1
    assert rag.sources() == {"keep.txt": 1}
    assert NaiveRAG(emb, index).sources() == {"keep.txt": 1}  # the removal was saved
    assert rag.remove_file("never-indexed.txt") == 0


def test_remove_everywhere_covers_every_index_of_that_user_only(tmp_path):
    doc = tmp_path / "shared.txt"
    doc.write_text("The retention period is 30 days.")
    emb = DeterministicFakeEmbedding(size=16)
    mine = [NaiveRAG(emb, Saved("me", name)) for name in ("naive_rag", "naive_rag.sentences")]
    theirs = NaiveRAG(emb, Saved("them", "naive_rag"))
    for rag in [*mine, theirs]:
        rag.add_file(doc)

    assert remove_everywhere("me", "shared.txt") == 2
    assert remove_everywhere("me", "shared.txt") == 0  # already gone: nothing rewritten
    for rag in mine:
        assert not NaiveRAG(emb, rag.saved).sources()
    assert NaiveRAG(emb, Saved("them", "naive_rag")).sources() == {"shared.txt": 1}
