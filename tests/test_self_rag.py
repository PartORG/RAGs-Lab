from langchain_core.language_models import FakeListChatModel

from rags.self_rag import SelfRAG


def test_rewrites_after_irrelevant_chunks_and_redrafts_an_ungrounded_answer(make_index):
    index = make_index("Boing is a remake of Pong.", "Chapter one.")
    replies = [
        "NO",  # not small talk, so the documents are used
        "NO",  # first chunks irrelevant
        "Boing is a remake of Pong.",  # rewritten query
        "YES",  # now relevant
        "Boing was made in 1972.",  # draft 1
        "NO",  # not grounded
        "Boing is a remake of Pong.",  # draft 2
        "YES",  # grounded
    ]
    result = SelfRAG(index, k=1).ask("what is boing", FakeListChatModel(responses=replies))
    assert result.answer == "Boing is a remake of Pong."
    assert result.chunks[0][0].page_content == "Boing is a remake of Pong."
    assert result.trace.splitlines() == [
        "Needs the documents? YES",
        "Searched for: what is boing",
        "Chunks relevant? NO",
        "Searched for: Boing is a remake of Pong.",
        "Chunks relevant? YES",
        "Draft 1 grounded? NO",
        "Draft 2 grounded? YES",
    ]


def test_small_talk_skips_retrieval(make_index):
    index = make_index("Boing is a remake of Pong.")
    result = SelfRAG(index).ask("hello!", FakeListChatModel(responses=["YES", "Hi there."]))
    assert result.answer == "Hi there." and result.chunks == []
    assert result.trace == "Small talk: answered without retrieval."
