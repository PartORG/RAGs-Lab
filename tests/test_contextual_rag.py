from langchain_core.language_models import FakeListChatModel

from rags.contextual_rag import ContextualRAG
from storage import Saved


class KeywordReranker:  # stands in for the cross-encoder
    def predict(self, pairs):
        return [float("platformer" in chunk) for _, chunk in pairs]


def test_headers_are_prepended_indexed_and_the_result_reranked(make_index, tmp_path):
    index = make_index("The bat moves.", "Cavern is a platformer.")
    rag = ContextualRAG(index, Saved("tester", "contextual"), KeywordReranker(), fetch_k=2, k=1)
    assert rag.pending() == 2
    rag.build(FakeListChatModel(responses=["From the book."]), lambda done, total: None)
    assert rag.pending() == 0
    assert {r["text"] for r in rag.store.store.values()} == {
        "From the book.\n\nThe bat moves.",
        "From the book.\n\nCavern is a platformer.",
    }

    result = rag.ask("which game is a platformer", FakeListChatModel(responses=["Cavern"]))
    assert [doc.page_content for doc, _ in result.chunks] == [
        "From the book.\n\nCavern is a platformer."
    ]
    assert result.rerank_scores == [1.0]
