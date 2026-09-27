from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from rags.self_query_rag import Search, SelfQueryRAG


class Parser(FakeListChatModel):
    """Returns a fixed parse of the question, as the real model's structured output would."""

    search: Search = Search(text="")

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda prompt: self.search)


def test_filters_keep_out_chunks_the_question_excluded(make_index, tmp_path):
    index = make_index("Boing is a remake of Pong.", "Cavern is a platformer.")
    parse = Search(text="games", source="1.txt")  # make_index names the second file 1.txt
    llm = Parser(responses=["Cavern."], search=parse)
    result = SelfQueryRAG(index).ask("what does 1.txt say about games?", llm)
    assert [doc.page_content for doc, _ in result.chunks] == ["Cavern is a platformer."]
    assert result.trace == "Searched for: 'games'\nFilters: file name contains 1.txt"


def test_a_filter_matching_nothing_is_reported(make_index):
    index = make_index("Boing is a remake of Pong.")
    llm = Parser(responses=["none"], search=Search(text="games", page_min=40))
    result = SelfQueryRAG(index).ask("what does page 40 say?", llm)
    assert result.chunks == []
    assert result.trace.endswith("No chunk passed the filters.")
