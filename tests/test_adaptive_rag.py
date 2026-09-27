from langchain_core.language_models import FakeListChatModel

from rags.adaptive_rag import AdaptiveRAG


def test_router_picks_the_path(make_index):
    index = make_index("Boing is a remake of Pong.")
    rag = AdaptiveRAG(index)

    direct = rag.ask("hello", FakeListChatModel(responses=["NONE", "Hi."]))
    assert direct.answer == "Hi." and direct.chunks == []
    assert direct.trace == "Router: NONE: answered directly, no retrieval"

    simple = rag.ask("what is boing", FakeListChatModel(responses=["SIMPLE", "A remake."]))
    assert simple.answer == "A remake." and len(simple.chunks) == 1
    assert simple.trace == "Router: SIMPLE: Naive RAG"

    llm = FakeListChatModel(responses=["COMPLEX", "DONE", "note", "DONE", "Combined."])
    complex_ = rag.ask("compare everything", llm)
    assert complex_.answer == "Combined."
    assert complex_.trace.startswith("Router: COMPLEX: Iterative / Multi-hop RAG\n\nQ: ")
