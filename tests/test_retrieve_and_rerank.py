from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from rags.naive_rag import NaiveRAG
from rags.retrieve_and_rerank import RerankRAG
from storage import Saved


class KeywordReranker:  # stands in for the cross-encoder
    def predict(self, pairs):
        return [float("30 days" in chunk) for _, chunk in pairs]


def test_reranker_not_vector_search_picks_what_reaches_the_llm(tmp_path):
    index = NaiveRAG(DeterministicFakeEmbedding(size=16), Saved("tester", "naive_rag"))
    for i, text in enumerate(["Pygame draws sprites.", "Chapter one.", "Retention is 30 days."]):
        (tmp_path / f"{i}.txt").write_text(text)
        index.add_file(tmp_path / f"{i}.txt")

    # Identical text makes "Pygame draws sprites." the vector-search winner; the reranker overrules.
    [top] = index.store.similarity_search("Pygame draws sprites.", k=1)
    assert top.page_content == "Pygame draws sprites."
    rag = RerankRAG(index, KeywordReranker(), fetch_k=3, k=1)
    result = rag.ask("Pygame draws sprites.", FakeListChatModel(responses=["30 days"]))

    assert [doc.page_content for doc, _ in result.chunks] == ["Retention is 30 days."]
    assert result.rerank_scores == [1.0]
    assert result.answer == "30 days"
