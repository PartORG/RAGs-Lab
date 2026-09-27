from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from rags.agentic_rag import AgenticRAG


class ToolCallingFake(GenericFakeChatModel):
    """Replies with the scripted messages; bind_tools is a no-op, as the tools are not really
    chosen by a model here."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_the_model_searches_twice_then_answers(make_index):
    index = make_index("Boing is a remake of Pong.", "Cavern is a platformer.")
    scripted = iter(
        [
            AIMessage(
                "", tool_calls=[{"name": "search_documents", "args": {"query": "boing"}, "id": "1"}]
            ),
            AIMessage(
                "",
                tool_calls=[{"name": "search_documents", "args": {"query": "cavern"}, "id": "2"}],
            ),
            AIMessage("Boing remakes Pong; Cavern is a platformer."),
        ]
    )
    result = AgenticRAG(index, k=1).ask("what are the games?", ToolCallingFake(messages=scripted))
    assert result.answer == "Boing remakes Pong; Cavern is a platformer."
    assert result.trace == "Searches the model chose:\n- boing\n- cavern"
    assert len(result.chunks) == 2  # one chunk from each search, kept in order


def test_answering_without_searching_is_allowed(make_index):
    index = make_index("Boing is a remake of Pong.")
    scripted = iter([AIMessage("Hello!")])
    result = AgenticRAG(index).ask("hello", ToolCallingFake(messages=scripted))
    assert result.answer == "Hello!" and result.chunks == []
    assert result.trace == "The model answered without searching."
