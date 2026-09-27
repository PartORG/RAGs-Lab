from langchain_core.language_models import FakeListChatModel

from rags.corrective_rag import CorrectiveRAG


def test_only_relevant_chunks_reach_the_answer(make_index):
    index = make_index("Boing is a remake of Pong.", "Chapter one.")
    llm = FakeListChatModel(responses=["YES", "NO", "Pong."])  # two grades, then the answer
    result = CorrectiveRAG(index, k=2).ask("Boing is a remake of Pong.", llm)
    assert [doc.page_content for doc, _ in result.chunks] == ["Boing is a remake of Pong."]
    assert result.answer == "Pong."
    assert result.trace == "Graded 2 retrieved chunks: 1 relevant."


def test_with_nothing_relevant_it_searches_again_then_refuses_to_guess(make_index):
    index = make_index("Boing is a remake of Pong.", "Chapter one.")
    # two failed grades, the rewritten query, two more failed grades; no answer is ever generated
    llm = FakeListChatModel(responses=["NO", "NO", "boing pong", "NO", "NO"])
    result = CorrectiveRAG(index, k=2, fallback_k=2).ask("Who made the original?", llm)
    assert result.answer == "I don't know based on the documents." and result.chunks == []
    assert "Rewrote the question as the query: boing pong" in result.trace
    assert result.trace.endswith("said so instead of answering from bad context.")
