from langchain_core.language_models import FakeListChatModel

from rags.hierarchical_rag import HierarchicalRAG
from storage import Saved


def test_children_match_parents_answer_and_a_reupload_rebuilds_only_that_file(make_index, tmp_path):
    index = make_index("The bat moves up and down. " * 20, "Cavern is a platformer.")
    rag = HierarchicalRAG(index, Saved("tester", "children"), k=1)
    assert rag.pending() == 2
    progress = []
    rag.build(FakeListChatModel(responses=[""]), lambda done, total: progress.append((done, total)))
    assert rag.pending() == 0 and progress[-1] == (2, 2)
    assert len(rag.children.store) > 2  # the long parent was split into several children

    result = rag.ask("Cavern is a platformer.", FakeListChatModel(responses=["ok"]))
    assert [doc.page_content for doc, _ in result.chunks] == ["Cavern is a platformer."]

    assert HierarchicalRAG(index, Saved("tester", "children")).pending() == 0  # saved
    index.add_file(tmp_path / "1.txt")  # re-upload: the Cavern chunk gets a new id
    assert rag.pending() == 1
    parents = {r["metadata"]["parent"] for r in rag.children.store.values()}
    assert parents <= index.store.store.keys()  # children of the replaced chunk were dropped
