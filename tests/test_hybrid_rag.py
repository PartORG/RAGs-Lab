from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from rags.hybrid_rag import BM25, HybridRAG, rrf


def test_rrf_favours_docs_ranked_high_in_several_lists():
    a, b, c = (Document(t, id=t) for t in "abc")
    assert [d.id for d in rrf([[a, b], [b, c]])] == ["b", "a", "c"]


def test_bm25_finds_the_exact_token_and_hybrid_passes_it_on(make_index):
    index = make_index("ERR_771 means the tracker lost power.", *[f"Note {i}." for i in range(12)])
    [top] = BM25(index.chunks()).search("what does ERR_771 mean", 1)
    assert top.page_content.startswith("ERR_771")

    result = HybridRAG(index, fetch_k=3).ask(
        "what does ERR_771 mean", FakeListChatModel(responses=["ok"])
    )
    assert any(doc.page_content.startswith("ERR_771") for doc, _ in result.chunks)
