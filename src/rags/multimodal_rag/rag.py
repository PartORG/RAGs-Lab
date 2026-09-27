"""Multimodal RAG: figures and screenshots become searchable through captions.

At build time a local vision model describes every sizeable image in the uploaded PDFs; the
captions are embedded and searched together with the Naive RAG text chunks. When a figure is
among the best hits, the vision model answers looking at the image itself, plus the text context.
Pages without text (scans, full-page art) are covered too: images are read from the PDF files.
"""

import hashlib
import io
import logging
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from pypdf import PageObject, PdfReader

from rags.naive_rag import NaiveRAG, Result, answer, context_of, graded
from storage import Saved

log = logging.getLogger(__name__)

CAPTION_PROMPT = """Describe this figure from page {page} of "{source}" in detail for search
indexing: what it shows, and any visible text, labels and numbers."""

# The shared ANSWER_PROMPT says to use only the text context, which makes the vision model ignore
# the very figures it was given; this one lets it read them.
VISION_ANSWER_PROMPT = """Answer the question using ONLY the context below and the attached figures.
If neither contains the answer, say "I don't know based on the documents."

Context:
{context}

Question: {question}"""

Vision = Callable[[str, list[bytes]], str]  # (prompt, PNG images) -> reply


def figures(page: PageObject, min_side: int = 150, max_side: int = 768) -> list[bytes]:
    """The page's images as PNG bytes: icons and decorations under min_side px are skipped, and
    big ones shrunk to max_side so a few fit in the vision model's context."""
    try:
        images = list(page.images)
    except Exception as e:  # a malformed page: no figures rather than a failed build
        log.warning("Could not read a page's images, skipped: %r", e)
        return []
    pngs = []
    for image in images:
        try:
            picture = image.image.convert("RGB")
        except Exception as e:  # pypdf cannot decode every image format
            log.debug("Could not decode an image, skipped: %r", e)
            continue
        if min(picture.size) < min_side:
            continue
        picture.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        picture.save(buffer, "PNG")
        pngs.append(buffer.getvalue())
    return pngs


class MultimodalRAG:
    unit = "pages"
    seconds_per_item = 1.8  # ~0.6 figures per page at ~2.7 s each: measured with qwen2.5vl:3b

    def __init__(self, index: NaiveRAG, saved: Saved, uploads: Path, vision: Vision, k: int = 4):
        self.index, self.saved, self.uploads, self.vision, self.k = index, saved, uploads, vision, k
        self.images = uploads.parent / "multimodal_rag_images"
        state = saved.load() or {"done": {}, "captions": {}}
        self.done: dict[str, dict] = state["done"]  # PDF name -> {"version", "pages" captioned}
        self.captions = InMemoryVectorStore(index.store.embeddings)
        self.captions.store = state["captions"]

    def _save(self) -> None:
        self.saved.save({"done": self.done, "captions": self.captions.store})

    def _version(self, name: str) -> str:
        stat = (self.uploads / name).stat()  # a re-upload rewrites the file
        return f"{stat.st_size}-{stat.st_mtime_ns}"

    def _todo(self) -> list[tuple[str, int]]:
        """(PDF, page) pairs not captioned yet; forgets the captions of re-uploaded files."""
        current = {p.name: self._version(p.name) for p in sorted(self.uploads.glob("*.pdf"))}
        stale = [name for name, entry in self.done.items() if current.get(name) != entry["version"]]
        if stale:
            for name in stale:
                del self.done[name]
            keep = {}
            for i, record in self.captions.store.items():
                if record["metadata"]["source"] in stale:
                    # the figure file too, or a deleted PDF leaves its images behind
                    (self.images / record["metadata"]["image"]).unlink(missing_ok=True)
                else:
                    keep[i] = record
            self.captions.store = keep
            self._save()
        todo = []
        for name in current:
            done = set(self.done.get(name, {}).get("pages", []))
            pages = len(PdfReader(self.uploads / name).pages)
            todo += [(name, page) for page in range(1, pages + 1) if page not in done]
        return todo

    def pending(self) -> int:
        return len(self._todo())

    def build(self, llm: BaseChatModel, progress: Callable[[int, int], None]) -> None:
        todo = self._todo()
        self.images.mkdir(parents=True, exist_ok=True)
        readers: dict[str, PdfReader] = {}
        for i, (name, page) in enumerate(todo, 1):
            reader = readers.setdefault(name, PdfReader(self.uploads / name))
            docs = []
            for png in figures(reader.pages[page - 1]):
                key = hashlib.sha1(png).hexdigest()[:16]
                if key in self.captions.store:  # the same figure on another page
                    continue
                (self.images / f"{key}.png").write_bytes(png)
                caption = self.vision(CAPTION_PROMPT.format(page=page, source=name), [png])
                meta = {"source": name, "page": page, "image": f"{key}.png"}
                docs.append(Document(caption, id=key, metadata=meta))
            if docs:
                self.captions.add_documents(docs)
            entry = self.done.setdefault(name, {"version": self._version(name), "pages": []})
            entry["pages"].append(page)
            if i % 5 == 0 or i == len(todo):  # save every 5 pages, so a build resumes
                self._save()
                progress(i, len(todo))

    def ask(self, question: str, llm: BaseChatModel) -> Result:
        texts = self.index.search(question, self.k)
        captions = self.captions.similarity_search_with_score(question, k=self.k)
        best = sorted(texts + captions, key=lambda t: t[1], reverse=True)[: self.k]
        # ponytail: at most 2 images, to stay inside Ollama's default 4096-token context.
        hits = [doc.metadata["image"] for doc, _ in best if "image" in doc.metadata]
        shown = [name for name in hits if (self.images / name).exists()][:2]  # deleted: text only
        if not shown:
            return answer(question, best, llm)
        prompt = VISION_ANSWER_PROMPT.format(context=context_of(best), question=question)
        reply = self.vision(prompt, [(self.images / name).read_bytes() for name in shown])
        result = graded(question, best, reply, llm)
        result.trace = f"Answered by the vision model, looking at {len(shown)} figure(s)."
        return result
