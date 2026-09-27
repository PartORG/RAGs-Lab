from langchain_core.language_models import FakeListChatModel

from rags.cag_rag import CAGRag


class CountingFake(FakeListChatModel):
    """FakeListChatModel with the model_copy(update=...) that CAG uses to widen the context."""

    num_ctx: int = 0
    keep_alive: str = ""


def test_the_whole_corpus_is_sent_when_it_fits(make_index):
    index = make_index("Boing is a remake of Pong.", "Cavern is a platformer.")
    llm = CountingFake(responses=["Both games."])
    result = CAGRag(index, num_ctx=4096).ask("which games?", llm)
    assert result.answer == "Both games."
    assert len(result.chunks) == 2  # every chunk was in the prompt, scored for the table
    assert result.trace.startswith("No retrieval: all 2 chunks")


def test_it_refuses_instead_of_truncating_a_corpus_that_does_not_fit(make_index):
    index = make_index("word " * 500)
    result = CAGRag(index, num_ctx=1600).ask("which games?", CountingFake(responses=["unused"]))
    assert result.answer.startswith("The documents are about")
    assert result.chunks == [] and result.trace == "Nothing was sent to the model."
