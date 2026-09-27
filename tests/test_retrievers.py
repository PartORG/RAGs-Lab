from langchain_core.language_models import FakeListChatModel

from retrievers import DEFAULT, RETRIEVERS
from retrievers.bm25 import bm25
from retrievers.hybrid import hybrid
from retrievers.mmr import mmr
from retrievers.reranked import reranked
from retrievers.similarity import similarity
from retrievers.threshold import threshold

NOTES = ["ERR_771 means the tracker lost power.", *[f"Note {i} about nothing." for i in range(8)]]


class KeywordReranker:  # stands in for the cross-encoder
    def predict(self, pairs):
        return [float("ERR_771" in chunk) for _, chunk in pairs]


def test_the_default_is_plain_similarity_and_always_returns_k(make_index):
    index = make_index(*NOTES)
    assert DEFAULT == "similarity"
    assert len(similarity(index)("what is ERR_771", 4)) == 4
    assert all(isinstance(score, float) for _, score in similarity(index)("anything", 3))


def test_the_floor_returns_fewer_rather_than_weak_chunks(make_index):
    index = make_index(*NOTES)
    question = NOTES[0]  # identical text scores 1.0 with the fake embeddings; the rest are noise
    kept = threshold(index, floor=0.99)(question, 4)
    assert [doc.page_content for doc, _ in kept] == [NOTES[0]]
    assert threshold(index, floor=1.01)(question, 4) == []  # nothing clears it: no context at all


def test_bm25_finds_the_exact_token_and_scores_it_by_cosine(make_index):
    index = make_index(*NOTES)
    [(doc, score)] = bm25(index)("what does ERR_771 mean", 1)
    assert doc.page_content.startswith("ERR_771")
    assert -1.0 <= score <= 1.0  # the cosine, not the BM25 score, so the UI stays comparable


def test_hybrid_merges_both_lists(make_index):
    index = make_index(*NOTES)
    found = [doc.page_content for doc, _ in hybrid(index)("what does ERR_771 mean", 4)]
    assert any(t.startswith("ERR_771") for t in found)  # the keyword side contributed
    assert len(found) == 4


def test_mmr_returns_k_without_repeating_a_chunk(make_index):
    index = make_index(*NOTES)
    found = [doc.page_content for doc, _ in mmr(index)("Note 3 about nothing.", 4)]
    assert len(found) == 4 and len(set(found)) == 4


def test_reranked_puts_the_cross_encoder_choice_first(make_index):
    index = make_index(*NOTES)
    found = reranked(index, lambda: KeywordReranker(), fetch=9)("tracker power", 2)
    assert found[0][0].page_content.startswith("ERR_771")


def test_the_chosen_retriever_reaches_a_strategy(make_index):
    """Naive RAG searches through NaiveRAG.search, so the sidebar's choice applies to it."""
    index = make_index(*NOTES)
    index.retriever = bm25(index)
    result = index.ask("what does ERR_771 mean", FakeListChatModel(responses=["lost power"]))
    assert result.chunks[0][0].page_content.startswith("ERR_771")


def test_every_registry_entry_builds_and_retrieves(make_index):
    index = make_index(*NOTES)
    for key, entry in RETRIEVERS.items():
        retrieve = entry.build(index, lambda: KeywordReranker())
        hits = retrieve("what does ERR_771 mean", 3)
        assert all(-1.0 <= score <= 1.0 for _, score in hits), key  # every one reports a cosine
        assert len(hits) <= 3, key
