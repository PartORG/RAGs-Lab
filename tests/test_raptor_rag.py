from langchain_core.language_models import FakeListChatModel

from rags.raptor_rag import RaptorRAG
from storage import Saved


class NoLLM:
    def invoke(self, prompt):
        raise AssertionError("saved summaries must be reused, not regenerated")


def test_tree_is_built_searched_resumed_and_goes_stale_on_upload(make_index, tmp_path):
    index = make_index(*[f"Fact number {i} about the games." for i in range(10)])
    rag = RaptorRAG(index, Saved("tester", "raptor"))
    assert rag.pending() == 10
    rag.build(FakeListChatModel(responses=["Summary of the games."]), lambda done, total: None)
    assert rag.pending() == 0
    assert len(rag.summaries.store) == 2  # 10 chunks -> 2 clusters; too few to cluster again

    result = rag.ask("Summary of the games.", FakeListChatModel(responses=["ok"]))
    assert result.chunks[0][0].metadata["source"] == "RAPTOR summary, level 1"

    reloaded = RaptorRAG(index, Saved("tester", "raptor"))
    reloaded.covers = set()  # as if the build had been interrupted after the last summary
    reloaded.build(NoLLM(), lambda done, total: None)
    assert reloaded.pending() == 0

    (tmp_path / "new.txt").write_text("A new fact.")
    index.add_file(tmp_path / "new.txt")
    assert rag.pending() == 11  # any upload makes the whole tree stale
