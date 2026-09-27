from langchain_core.language_models import FakeListChatModel
from PIL import Image

from rags.multimodal_rag import MultimodalRAG
from storage import Saved


def test_figures_are_captioned_found_shown_to_the_model_and_forgotten_on_delete(
    make_index, tmp_path
):
    index = make_index("Chapter one.")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    Image.new("RGB", (300, 200), "red").save(uploads / "fig.pdf")  # one page, one image
    Image.new("RGB", (40, 40), "blue").save(uploads / "icon.pdf")  # too small to caption

    seen = []

    def vision(prompt: str, images: list[bytes]) -> str:
        seen.append(len(images))
        return "A red rectangle." if prompt.startswith("Describe") else "It is red."

    rag = MultimodalRAG(index, Saved("tester", "multimodal"), uploads, vision)
    assert rag.pending() == 2  # pages: one per PDF
    rag.build(FakeListChatModel(responses=[""]), lambda done, total: None)
    assert rag.pending() == 0 and seen == [1]  # only the big image was captioned
    [record] = rag.captions.store.values()
    assert record["metadata"]["page"] == 1
    assert (tmp_path / "multimodal_rag_images" / record["metadata"]["image"]).exists()

    result = rag.ask("A red rectangle.", FakeListChatModel(responses=["unused"]))
    assert result.answer == "It is red." and seen[-1] == 1  # answered looking at the figure

    image = tmp_path / "multimodal_rag_images" / record["metadata"]["image"]
    (uploads / "fig.pdf").unlink()  # the user deletes it from the sidebar
    assert rag.pending() == 0  # nothing left to caption: the other PDF was done already
    assert not rag.captions.store and not image.exists()  # caption and figure file both gone
