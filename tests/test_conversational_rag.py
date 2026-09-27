from langchain_core.language_models import FakeListChatModel

from rags.conversational_rag import ConversationalRAG


def test_follow_up_is_condensed_before_search_but_first_turn_is_not(make_index):
    index = make_index("Boing is a remake of Pong.", "Cavern is a platformer.")
    first = ConversationalRAG(index, []).ask(
        "What is Boing?", FakeListChatModel(responses=["Pong"])
    )
    assert first.answer == "Pong" and not first.trace  # the only LLM call was the answer

    llm = FakeListChatModel(responses=["Cavern is a platformer.", "A platformer."])
    rag = ConversationalRAG(index, [("What is Boing?", "Pong")])
    follow_up = rag.ask("And the other game?", llm)
    assert follow_up.trace == "Standalone question: Cavern is a platformer."
    assert follow_up.chunks[0][0].page_content == "Cavern is a platformer."  # searched rewritten
    assert follow_up.answer == "A platformer."
