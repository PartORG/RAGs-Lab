from langchain_core.language_models import FakeListChatModel

from rags.branched_rag import BranchedRAG


def test_every_branch_is_asked_and_the_rankings_merged(make_index):
    index = make_index("ERR_771 means the tracker lost power.", *[f"Note {i}." for i in range(8)])
    extra = [("summaries", lambda q, n: index.store.similarity_search("Note 3.", k=1))]
    rag = BranchedRAG(index, extra, per_branch=2, k=4)
    result = rag.ask("what does ERR_771 mean", FakeListChatModel(responses=["ok"]))
    texts = [doc.page_content for doc, _ in result.chunks]
    assert any(t.startswith("ERR_771") for t in texts)  # the keyword branch found it
    assert "Note 3." in texts  # the extra branch contributed too
    assert result.trace.startswith("Asked every source at once:\n- text chunks: 2 passages")
    assert "- summaries: 1 passages" in result.trace
