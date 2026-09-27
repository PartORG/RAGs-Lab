from langchain_core.language_models import FakeListChatModel

from rags.multi_hop_rag import MultiHopRAG


def test_each_hop_shapes_the_next_until_done_then_synthesizes(make_index):
    index = make_index("Boing is a remake of Pong.", "Pong was made by Atari.")
    replies = [
        "Boing is a remake of Pong.",  # sub-question 1: its exact text finds that chunk
        "Pong",  # hop 1 note
        "Pong was made by Atari.",  # sub-question 2
        "Atari",  # hop 2 note
        "DONE",
        "Atari made the game Boing remakes.",  # synthesis
    ]
    rag = MultiHopRAG(index, k=1)
    result = rag.ask("Who made the game Boing remakes?", FakeListChatModel(responses=replies))
    assert result.answer == "Atari made the game Boing remakes."
    assert result.trace == (
        "Q: Boing is a remake of Pong.\nA: Pong\n\nQ: Pong was made by Atari.\nA: Atari"
    )
    assert {doc.page_content for doc, _ in result.chunks} == {
        "Boing is a remake of Pong.",
        "Pong was made by Atari.",
    }


def test_done_before_any_lookup_still_searches_the_question(make_index):
    index = make_index("Pong was made by Atari.")
    llm = FakeListChatModel(responses=["DONE", "Atari", "DONE", "Atari."])
    result = MultiHopRAG(index, k=1).ask("Pong was made by Atari.", llm)
    assert result.trace == "Q: Pong was made by Atari.\nA: Atari"
