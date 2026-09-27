from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from rags.graph_rag import Facts, GraphRAG
from storage import Saved


class Extractor(FakeListChatModel):
    """Returns the same facts for every chunk it is asked to extract."""

    facts: list[tuple[str, str, str]] = []

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda prompt: Facts(facts=self.facts))


def test_facts_are_extracted_then_walked_outward_from_the_question_entities(make_index, tmp_path):
    index = make_index("Boing is a remake of Pong, which Atari made.")
    rag = GraphRAG(index, Saved("tester", "graph"))
    facts = [
        ("pong", "was made by", "Atari"),
        ("Boing", "is a remake of", "pong"),
        ("nibbles", "is", "unrelated"),
    ]
    rag.build(Extractor(responses=[""], facts=facts), lambda done, total: None)
    assert rag.pending() == 0

    llm = FakeListChatModel(responses=["boing", "Atari."])  # the entities, then the answer
    result = rag.ask("Who made the game Boing remakes?", llm)
    assert [doc.page_content for doc, _ in result.chunks] == [
        "boing -[is a remake of]-> pong",  # 1 hop from "boing" beats 3 hops
        "pong -[was made by]-> atari",
    ]
    assert result.chunks[0][0].metadata["source"] == "0.txt"  # traceable to its chunk
    assert result.answer == "Atari."
