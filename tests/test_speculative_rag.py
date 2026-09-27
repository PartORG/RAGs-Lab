from langchain_core.language_models import FakeListChatModel

from rags.speculative_rag import SpeculativeRAG


def test_the_verifier_picks_among_one_draft_per_chunk(make_index):
    index = make_index("Boing is a remake of Pong.", "Cavern is a platformer.")
    drafter = FakeListChatModel(responses=["draft zero", "draft one"])
    verifier = FakeListChatModel(responses=["[1]"])  # picks the second draft
    result = SpeculativeRAG(index, drafter, k=2).ask("what is boing", verifier)
    assert result.answer == "draft one"
    assert result.trace.startswith("The verifier picked draft [1] of 2.")
    assert "[0] draft zero" in result.trace


def test_an_unparsable_or_out_of_range_pick_falls_back_to_the_first_draft(make_index):
    index = make_index("Boing is a remake of Pong.")
    drafter = FakeListChatModel(responses=["only draft"])
    result = SpeculativeRAG(index, drafter, k=1).ask("what", FakeListChatModel(responses=["七"]))
    assert result.answer == "only draft"
