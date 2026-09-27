"""Upload documents, chat with them through a chosen RAG strategy, and see how good each
answer was."""

import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import ollama
import streamlit as st
from langchain_ollama import ChatOllama
from pypdf import PdfReader

from chunking_strategies import CHUNKERS
from chunking_strategies import DEFAULT as DEFAULT_CHUNKING
from docs import mermaid_to_dot, section
from embeddings import DEFAULT as DEFAULT_EMBEDDING
from embeddings import EMBEDDINGS
from rags.adaptive_rag import AdaptiveRAG
from rags.agentic_rag import AgenticRAG
from rags.branched_rag import BranchedRAG
from rags.cag_rag import CAGRag
from rags.contextual_rag import ContextualRAG
from rags.conversational_rag import ConversationalRAG
from rags.corrective_rag import CorrectiveRAG
from rags.graph_rag import GraphRAG
from rags.hierarchical_rag import HierarchicalRAG
from rags.hybrid_rag import HybridRAG
from rags.hyde_rag import HydeRAG
from rags.multi_hop_rag import MultiHopRAG
from rags.multi_query_rag import MultiQueryRAG
from rags.multimodal_rag import MultimodalRAG
from rags.naive_rag import NaiveRAG, Result, remove_everywhere
from rags.raptor_rag import RaptorRAG
from rags.retrieve_and_rerank import RerankRAG
from rags.self_query_rag import SelfQueryRAG
from rags.self_rag import SelfRAG
from rags.speculative_rag import SpeculativeRAG
from retrievers import RETRIEVERS
from storage import Saved, account_stamp, check_password, user_dir

# Settings come from the environment (see README); the Ollama address is OLLAMA_HOST, which the
# ollama client reads itself.
DEFAULT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen3:8b")
# Hugging Face cross-encoder. Not bge-reranker-v2-m3: its ~3 GB in RAM next to Ollama ran this
# 14 GB laptop out of memory. This one is half the size, but mainly English.
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
# Multimodal RAG only. The 3b, not the 7b: the 7b needs as much memory again as the chat model.
VISION_MODEL = os.environ.get("RAG_VISION_MODEL", "qwen2.5vl:3b")
# Speculative RAG drafts with this, if it is installed, and verifies with the sidebar model.
DRAFT_MODEL = os.environ.get("RAG_DRAFT_MODEL", "llama3.2:3b")

# One line per event on stdout, where Docker collects it. basicConfig does nothing on reruns.
logging.basicConfig(
    level=os.environ.get("RAG_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # otherwise a line per Ollama request
log = logging.getLogger("rag_lab")

# Streamlit's file watcher probes every loaded module; transformers' lazy modules answer by
# importing optional vision code, which logs ~150 harmless torchvision tracebacks per rerun.
# Drop this line once a Streamlit or transformers upgrade stops it.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)


def chunking_llm() -> ChatOllama:
    """The LLM-based chunkers run at indexing time on the default chat model, not the one picked
    for chatting: an index must not change meaning because a dropdown moved."""
    name = DEFAULT_MODEL if DEFAULT_MODEL in chat_models else chat_models[0]
    return ChatOllama(model=name, temperature=0, reasoning=False, num_predict=800)


def saved(user: str, name: str, chunking: str, embedding: str) -> Saved:
    """One set of indexes per user, chunking and embedding, so switching either throws nothing
    away. An index belongs to the model that built it: the query must be embedded the same way.
    The defaults keep the plain names."""
    parts = [
        part
        for part, default in ((chunking, DEFAULT_CHUNKING), (embedding, DEFAULT_EMBEDDING))
        if part != default
    ]
    return Saved(user, name + "".join(f".{part}" for part in parts))


def uploads_of(user: str) -> Path:
    return user_dir(user) / "uploads"


@contextmanager
def timed(event: str, **fields):
    """Log an action with its duration and outcome, one line: `question ok user=ann ... 4.2s`.
    A failure is logged with its traceback and shown to the user in one line, and the page
    carries on; the body can add fields (scores, counts) to the yielded dict."""
    start = time.perf_counter()
    try:
        yield fields
    except Exception as e:
        log.exception("%s failed %s %.1fs", event, _kv(fields), time.perf_counter() - start)
        fields["failed"] = True  # read by the caller, e.g. to keep the error on screen
        st.error(f"{event.capitalize()} failed: {type(e).__name__}: {e}. Details are in the log.")
    else:
        log.info("%s ok %s %.1fs", event, _kv(fields), time.perf_counter() - start)


def _kv(fields: dict) -> str:
    return " ".join(f"{key}={value!r}" for key, value in fields.items())


def login() -> None:
    """The gate in front of everything: accounts are made with `rag-lab adduser NAME`."""
    with st.form("login"):
        name = st.text_input("User name").strip()
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            if check_password(name, password):
                log.info("login ok user=%r", name)
                st.session_state.user = name
                st.session_state.stamp = account_stamp(name)
                st.rerun()
            log.warning("login failed user=%r", name)
            # ponytail: a pause per failed try, per session; a lockout table if the lab is ever
            # exposed beyond a trusted network.
            time.sleep(1)
            st.error("Wrong user name or password.")
    st.caption("No account yet? The admin creates one with `rag-lab adduser NAME`.")
    st.stop()


# Cached per user, chunking and embedding: every tab of one user shares the same index objects.
# `stamp` is only part of the cache key: see storage.account_stamp.
# ponytail: kept for the process lifetime, so memory grows with the users who logged in; add
# ttl= to these caches if that ever matters.
@st.cache_resource
def get_rag(user: str, chunking: str, embedding: str, stamp: str) -> NaiveRAG:
    vectors = EMBEDDINGS[embedding].build()
    chunker = CHUNKERS[chunking].build(vectors, chunking_llm())
    return NaiveRAG(vectors, saved(user, "naive_rag", chunking, embedding), chunker=chunker)


@st.cache_resource
def get_reranker():
    from sentence_transformers import CrossEncoder  # loads torch, so only once rerank is picked

    # ponytail: CPU, since Ollama's chat model already fills most of the GPU. Use device="cuda"
    # when there is VRAM to spare.
    return CrossEncoder(RERANK_MODEL, device="cpu")


# The strategies with an index of their own, built from the Naive RAG chunks: loaded once.
@st.cache_resource
def get_hierarchical(user: str, chunking: str, embedding: str, stamp: str) -> HierarchicalRAG:
    return HierarchicalRAG(
        get_rag(user, chunking, embedding, stamp),
        saved(user, "hierarchical_rag", chunking, embedding),
    )


@st.cache_resource
def get_raptor(user: str, chunking: str, embedding: str, stamp: str) -> RaptorRAG:
    return RaptorRAG(
        get_rag(user, chunking, embedding, stamp), saved(user, "raptor_rag", chunking, embedding)
    )


@st.cache_resource
def get_contextual(user: str, chunking: str, embedding: str, stamp: str) -> ContextualRAG:
    return ContextualRAG(
        get_rag(user, chunking, embedding, stamp),
        saved(user, "contextual_rag", chunking, embedding),
        get_reranker(),
    )


@st.cache_resource
def get_graph(user: str, chunking: str, embedding: str, stamp: str) -> GraphRAG:
    return GraphRAG(
        get_rag(user, chunking, embedding, stamp), saved(user, "graph_rag", chunking, embedding)
    )


def get_drafter() -> ChatOllama:
    """Speculative RAG's small drafter; the sidebar model itself when it is not installed."""
    name = DRAFT_MODEL if DRAFT_MODEL in chat_models else model
    return ChatOllama(model=name, temperature=0.7, reasoning=False, num_predict=300)


def see(prompt: str, images: list[bytes]) -> str:
    message = {"role": "user", "content": prompt, "images": images}
    return ollama.chat(VISION_MODEL, [message], options={"temperature": 0}).message.content


@st.cache_resource
def get_multimodal(user: str, chunking: str, embedding: str, stamp: str) -> MultimodalRAG:
    return MultimodalRAG(
        get_rag(user, chunking, embedding, stamp),
        saved(user, "multimodal_rag", chunking, embedding),
        uploads_of(user),
        see,
    )


def extra_branches(
    user: str, chunking: str, embedding: str, stamp: str
) -> list[tuple[str, object]]:
    """The other indexes Branched RAG fans out to, when they have been built."""
    branches = []
    raptor = get_raptor(user, chunking, embedding, stamp)
    if raptor.summaries.store:
        branches.append(
            ("RAPTOR summaries", lambda q, n: raptor.summaries.similarity_search(q, k=n))
        )
    figures = get_multimodal(user, chunking, embedding, stamp)
    if figures.captions.store:
        branches.append(
            ("figure captions", lambda q, n: figures.captions.similarity_search(q, k=n))
        )
    return branches


def page_count(path: Path) -> int:
    """Pages of a PDF, or one for a text file: what the indexing estimate is counted in."""
    if path.suffix.lower() != ".pdf":
        return 1
    try:
        return len(PdfReader(path).pages)
    except Exception as e:  # only an estimate: a PDF pypdf cannot open counts as one page
        log.warning("Could not count the pages of %s: %r", path.name, e)
        return 1


def chat_so_far() -> list[tuple[str, str]]:
    return [(question, result.answer) for question, _, result in st.session_state.history]


# Every strategy starts from the Naive RAG chunks, so one upload serves all of them. Hierarchical,
# RAPTOR, Contextual, GraphRAG and Multimodal also build an index of their own (sidebar button).
STRATEGIES = {
    "Naive RAG": lambda: get_rag(user, chunking, embedding, stamp),
    "Retrieve-and-Rerank": lambda: RerankRAG(
        get_rag(user, chunking, embedding, stamp), get_reranker()
    ),
    "Hybrid RAG": lambda: HybridRAG(get_rag(user, chunking, embedding, stamp)),
    "Multi-Query / RAG-Fusion": lambda: MultiQueryRAG(get_rag(user, chunking, embedding, stamp)),
    "HyDE": lambda: HydeRAG(get_rag(user, chunking, embedding, stamp)),
    "Conversational RAG": lambda: ConversationalRAG(
        get_rag(user, chunking, embedding, stamp), chat_so_far()
    ),
    "Hierarchical RAG": lambda: get_hierarchical(user, chunking, embedding, stamp),
    "RAPTOR": lambda: get_raptor(user, chunking, embedding, stamp),
    "Contextual RAG": lambda: get_contextual(user, chunking, embedding, stamp),
    "GraphRAG": lambda: get_graph(user, chunking, embedding, stamp),
    "Multimodal RAG": lambda: get_multimodal(user, chunking, embedding, stamp),
    "Corrective RAG (CRAG)": lambda: CorrectiveRAG(
        get_rag(user, chunking, embedding, stamp), web=st.session_state.get("web", False)
    ),
    "Self-RAG": lambda: SelfRAG(get_rag(user, chunking, embedding, stamp)),
    "Adaptive RAG": lambda: AdaptiveRAG(get_rag(user, chunking, embedding, stamp)),
    "Iterative / Multi-hop RAG": lambda: MultiHopRAG(get_rag(user, chunking, embedding, stamp)),
    "Speculative RAG": lambda: SpeculativeRAG(
        get_rag(user, chunking, embedding, stamp), get_drafter()
    ),
    "Agentic RAG": lambda: AgenticRAG(get_rag(user, chunking, embedding, stamp)),
    "Branched / Multi-Source RAG": lambda: BranchedRAG(
        get_rag(user, chunking, embedding, stamp), extra_branches(user, chunking, embedding, stamp)
    ),
    "Cache-Augmented Generation (CAG)": lambda: CAGRag(get_rag(user, chunking, embedding, stamp)),
    "Self-Query RAG": lambda: SelfQueryRAG(get_rag(user, chunking, embedding, stamp)),
}


# One icon and one colour per strategy. The colour never carries the meaning on its own: the icon
# and the name are always beside it, because 20 hues cannot all be told apart by a colour-blind
# reader. Generated in OKLCH at even lightness and chroma, ordered so neighbours in this list sit
# ~126 degrees apart, and validated (lightness band, chroma, contrast, normal-vision separation
# all pass in light and dark). Blue and the person icon belong to the user, never to a strategy.
USER_ICON, USER_COLOUR = "👤", "#4885df"
BADGES = {
    "Naive RAG": ("🔎", "#cc5a82"),
    "Retrieve-and-Rerank": ("🎯", "#a18400"),
    "Hybrid RAG": ("⚗️", "#00a19d"),
    "Multi-Query / RAG-Fusion": ("🔀", "#d05a6e"),
    "HyDE": ("🎭", "#908b00"),
    "Conversational RAG": ("💬", "#009cb7"),
    "Hierarchical RAG": ("🪆", "#d15b58"),
    "RAPTOR": ("🌲", "#709517"),
    "Contextual RAG": ("🏷️", "#956ed2"),
    "GraphRAG": ("🕸️", "#ce6234"),
    "Multimodal RAG": ("🖼️", "#549a39"),
    "Corrective RAG (CRAG)": ("🩹", "#a468c7"),
    "Self-RAG": ("🪞", "#c96811"),
    "Adaptive RAG": ("🚦", "#2e9e52"),
    "Iterative / Multi-hop RAG": ("🧭", "#b761b1"),
    "Speculative RAG": ("⚡", "#c17000"),
    "Agentic RAG": ("🤖", "#00a274"),
    "Branched / Multi-Source RAG": ("📡", "#c15d9f"),
    "Cache-Augmented Generation (CAG)": ("🧠", "#b07c00"),
    "Self-Query RAG": ("🔖", "#00a28a"),
}

STYLE = """<style>
/* The evaluation strip: a fixed slice of the window, so the chat above it never shrinks and the
   strip never grows. Inherited ink for text, the strategy's colour only on its badge. */
.rag-eval { height: 15vh; box-sizing: border-box; display: flex; align-items: center; gap: 1.6rem;
  padding: 0.6rem 0.2rem; border-top: 1px solid rgba(128,128,128,0.25); overflow: hidden; }
.rag-eval .badge { font-size: 0.85rem; font-weight: 600; white-space: nowrap; }
.rag-eval .score { display: flex; flex-direction: column; gap: 0.1rem; }
.rag-eval .value { font-size: 1.3rem; font-weight: 600; line-height: 1.1; }
.rag-eval .label { font-size: 0.68rem; letter-spacing: 0.04em; text-transform: uppercase;
  opacity: 0.6; white-space: nowrap; }
.rag-eval .judge { flex: 1; min-width: 0; font-size: 0.78rem; opacity: 0.7; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
.rag-eval .empty { opacity: 0.55; font-size: 0.85rem; }
/* The person's own colour, on their avatar: blue belongs to them, never to a strategy. The
   avatar div carries no testid of its own, so the message is found by its aria-label. */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:first-child {
  background: #4885df; border-color: #4885df; }
/* The ℹ️ buttons beside the dropdowns (keyed "info …"): no padding, so the icon fits its narrow
   column in a narrow sidebar instead of spilling past it; bordered in the accent blue. */
[class*="st-key-info-"] button { width: 100%; min-width: 0; padding: 0; border-color: #4885df; }
/* Trim the padding around the pinned input, so the chat keeps the room the strip does not use. */
[data-testid="stBottomBlockContainer"] { padding-top: 0.6rem; padding-bottom: 0.2rem; }
</style>"""


def score_html(value: str, label: str) -> str:
    return (
        f'<div class="score"><div class="value">{value}</div><div class="label">{label}</div></div>'
    )


def evaluation_strip(result: Result | None, strategy: str, note: str = "") -> None:
    """The static bottom element: how the last answer scored, and nothing else. Its details live
    in the chat message, so this stays one glance tall. It keeps its place while an answer is
    being generated, so the page never jumps."""
    if result is None:
        empty = note or (
            "The evaluation of an answer appears here: how well it answers, how grounded it is, "
            "and how much was found."
        )
        st.html(f'<div class="rag-eval"><span class="empty">{empty}</span></div>')
        return
    icon, colour = BADGES[strategy]
    scores = [score for _, score in result.chunks]
    grade = result.grade
    parts = [
        f'<span class="badge" style="color:{colour}">{icon} {strategy}</span>',
        score_html(f"{grade.relevance}/5" if grade else "–", "answers the question"),
        score_html(f"{grade.groundedness}/5" if grade else "–", "grounded in the chunks"),
        score_html(str(len(result.chunks)), "chunks found"),
        score_html(f"{max(scores):.2f}" if scores else "–", "best match"),
    ]
    judge = (
        f"Judge: {grade.reason} Graded by the model that answered, so treat it as a self-check."
        if grade
        else f"The model could not grade its answer ({result.grade_error})."
    )
    parts.append(f'<div class="judge">{judge}</div>')
    st.html(f'<div class="rag-eval">{"".join(parts)}</div>')


def answer_details(result: Result, strategy: str) -> None:
    """Under each answer in the chat: what the strategy did, and what it read."""
    icon, colour = BADGES[strategy]
    st.html(f'<span class="badge" style="color:{colour};font-size:0.8rem">{icon} {strategy}</span>')
    if result.trace:
        st.expander("What the strategy did").text(result.trace)
    if not result.chunks:
        return
    no_rerank = [None] * len(result.chunks)
    st.expander(f"Retrieved chunks ({len(result.chunks)})").dataframe(
        [
            {
                "rerank": rerank,
                "similarity": score,
                "source": doc.metadata["source"],
                "page": doc.metadata.get("page"),
                "chunk": doc.page_content,
            }
            for (doc, score), rerank in zip(
                result.chunks, result.rerank_scores or no_rerank, strict=True
            )
        ],
        column_config={
            "rerank": st.column_config.ProgressColumn(
                "rerank",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                help="Cross-encoder relevance of the question and chunk read together.",
            )
            if result.rerank_scores
            else None,  # None hides the column
            "similarity": st.column_config.ProgressColumn(
                "similarity", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
        hide_index=True,
        width="stretch",
    )


def with_info(guide: str, pick):
    """A dropdown (or any element `pick` draws) with an ℹ️ button beside it that opens its guide
    in the main area."""
    left, right = st.columns([5, 1], gap="small", vertical_alignment="bottom")
    with left:
        choice = pick()
    right.button(
        "ℹ️",
        key=f"info {guide}",
        help="How this works, pros and cons",
        on_click=lambda: st.session_state.update(info=guide),
    )
    return choice


def info_page(guide: str, key: str | None) -> None:
    """The guide's section for the current choice, its Mermaid diagrams drawn as graphs. It
    follows the dropdown, so switching the choice while reading shows the new one."""
    st.button("← Back to chat", on_click=lambda: st.session_state.update(info=None))
    parts = re.split(r"```mermaid\n(.*?)```", section(guide, key), flags=re.S)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            st.markdown(part)
        elif dot := mermaid_to_dot(part):
            st.graphviz_chart(dot)
        else:
            st.code(part, language="mermaid")


st.set_page_config(page_title="RAG Lab", page_icon="🔎", layout="wide")
st.html(STYLE)
st.title("RAG Lab")

if "user" not in st.session_state:
    login()
user = st.session_state.user
stamp = account_stamp(user)
if stamp != st.session_state.get("stamp"):
    # Deleted with `deluser`, or its password reset with `adduser`, while this tab was open.
    log.info("session ended user=%r: account deleted or password changed", user)
    st.session_state.clear()
    st.warning("Your account was changed or deleted: log in again.")
    login()
UPLOADS = uploads_of(user)

try:
    installed = [m.model for m in ollama.list().models]
except ConnectionError as e:
    log.error("Ollama is not reachable: %s", e)
    st.error("Ollama is not reachable. Start it (`ollama serve`) and reload the page.")
    st.stop()
chat_models = [m for m in installed if "embed" not in m]
if not chat_models:
    st.error(f"Pull a chat model first: `ollama pull {DEFAULT_MODEL}`.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []  # (question, strategy, Result)

with st.sidebar:
    who = st.columns([3, 2], vertical_alignment="center")
    who[0].markdown(f"👤 **{user}**")
    if who[1].button("Log out"):
        log.info("logout user=%r", user)
        st.session_state.clear()
        st.rerun()
    strategy = with_info("RAGS.md", lambda: st.selectbox("RAG strategy", list(STRATEGIES)))
    model = st.selectbox(
        "Ollama model",
        chat_models,
        index=chat_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in chat_models else 0,
    )
    # num_predict caps a runaway answer: GraphRAG's terse fact lists once sent qwen3 into a
    # repetition loop that ran for 4 minutes and 49,000 characters. Normal answers are far shorter.
    llm = ChatOllama(model=model, temperature=0, reasoning=False, num_predict=800)
    # No dropdown to follow here: the ℹ️ opens the whole guide to evaluating RAG.
    with_info(
        "EVALUATION_STRATEGIES.md",
        lambda: st.caption("Answers are self-graded in the bottom strip. Other ways to evaluate:"),
    )
    if strategy == "Corrective RAG (CRAG)":
        st.checkbox(
            "Allow web search fallback",
            key="web",
            help="When nothing in your documents is relevant, send the rewritten question — never "
            "the documents themselves — to DuckDuckGo. Off by default: this is the only thing in "
            "the lab that leaves the machine.",
        )
    if strategy == "Speculative RAG" and DRAFT_MODEL not in chat_models:
        st.caption(f"`{DRAFT_MODEL}` is not installed: drafting with {model} instead.")
    build_box = st.container()  # filled below, once this run's uploads are indexed
    st.header("Documents")
    embedding = with_info(
        "EMBEDDINGS.md",
        lambda: st.selectbox(
            "Embedding", list(EMBEDDINGS), format_func=lambda key: EMBEDDINGS[key].label
        ),
    )
    st.caption(EMBEDDINGS[embedding].describe)
    served_by_ollama = EMBEDDINGS[embedding].ollama
    if served_by_ollama and not any(m.startswith(served_by_ollama) for m in installed):
        st.error(f"Not installed: run `ollama pull {served_by_ollama}` and reload.")
        st.stop()
    chunking = with_info(
        "CHUNKING_STRATEGIES.md",
        lambda: st.selectbox(
            "Chunking", list(CHUNKERS), format_func=lambda key: CHUNKERS[key].label
        ),
    )
    st.caption(CHUNKERS[chunking].describe)
    retriever = with_info(
        "RETRIEVERS.md",
        lambda: st.selectbox(
            "Retriever", list(RETRIEVERS), format_func=lambda key: RETRIEVERS[key].label
        ),
    )
    st.caption(RETRIEVERS[retriever].describe)
    # An index belongs to one embedding and one chunking: each pair keeps its own files. The
    # retriever changes nothing on disk, so it is set on the loaded index for this run.
    rag = get_rag(user, chunking, embedding, stamp)
    rag.retriever = RETRIEVERS[retriever].build(rag, get_reranker)
    index_box = st.container()  # filled below: the files this chunking has not indexed yet
    files = st.file_uploader(
        "PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if files and st.button("Save & index", type="primary"):
        UPLOADS.mkdir(parents=True, exist_ok=True)
        for f in files:
            path = UPLOADS / Path(f.name).name  # basename only: the browser controls this string
            with (
                st.spinner(f"Indexing {path.name}…"),
                timed(
                    "index", user=user, file=path.name, chunking=chunking, embedding=embedding
                ) as fields,
            ):
                path.write_bytes(f.getvalue())
                fields["chunks"] = n = rag.add_file(path)
                if n:
                    st.success(f"{path.name}: {n} chunks")
                else:
                    st.warning(f"{path.name}: no text found (scanned PDF?)")
    st.caption("Your documents are private to your account.")
    chunks_of = rag.sources()
    # Files with no text (a scanned PDF) have no chunks but are still stored, and Multimodal RAG
    # reads their figures, so the list comes from the folder, not from the index.
    stored = sorted({p.name for p in UPLOADS.glob("*") if p.is_file()} | set(chunks_of))
    for name in stored:
        n = chunks_of.get(name, 0)
        chunks = f"{n} chunk" + ("" if n == 1 else "s")
        row = st.columns([5, 1], vertical_alignment="center")
        row[0].markdown(f"📄 {name} — {chunks}")
        with row[1].popover("✕", help=f"Delete {name}"):
            st.markdown(
                f"Delete **{name}**? That removes {chunks} from the index and the file from "
                "disk, and every strategy drops what it built from them."
            )
            if st.button("Delete", key=f"delete {name}", type="primary"):
                with timed("delete", user=user, file=name) as fields:
                    # The file is gone: none of this user's chunking x embedding indexes should
                    # still hold it. Edited in the database, so no embedding model is loaded.
                    fields["chunks"] = removed = remove_everywhere(user, name)
                    # Cached indexes still hold the old chunks: reload them. The strategy
                    # indexes then drop what they built from those chunks on their next look.
                    # ponytail: clears every user's cache, not just this one's; they reload
                    # from the database on their next click.
                    for cached in (
                        get_rag,
                        get_hierarchical,
                        get_raptor,
                        get_contextual,
                        get_graph,
                        get_multimodal,
                    ):
                        cached.clear()
                    (UPLOADS / name).unlink(missing_ok=True)
                    st.toast(f"Deleted {name} ({removed} chunk" + ("s" * (removed != 1)) + ")")
                if not fields.get("failed"):
                    st.rerun()

unindexed = [name for name in stored if name not in chunks_of]
if unindexed:
    pages = sum(page_count(UPLOADS / name) for name in unindexed)
    minutes = pages * CHUNKERS[chunking].seconds_per_page / 60
    cost = f", roughly {max(1, round(minutes))} min" if minutes > 1 else ""
    index_box.warning(
        f"{CHUNKERS[chunking].label} has not indexed {len(unindexed)} of these files yet: "
        f"{pages} pages{cost}."
    )
    if index_box.button(f"Index {len(unindexed)} file(s) with this chunking", type="primary"):
        bar = index_box.progress(0.0, "Starting…")
        failed = False
        for done, name in enumerate(unindexed, 1):
            with timed(
                "index", user=user, file=name, chunking=chunking, embedding=embedding
            ) as fields:
                fields["chunks"] = rag.add_file(UPLOADS / name)
            failed |= fields.get("failed", False)
            bar.progress(done / len(unindexed), f"{done} / {len(unindexed)}")
        if not failed:  # a rerun would wipe the error off the screen
            st.rerun()

pipeline = STRATEGIES[strategy]()
pending = pipeline.pending() if hasattr(pipeline, "pending") else 0
if pending:
    minutes = max(1, round(pending * pipeline.seconds_per_item / 60))
    unit = getattr(pipeline, "unit", "chunks")
    build_box.warning(
        f"{strategy} needs an index of its own: {pending} {unit} to process, roughly {minutes} min "
        "(as measured on this laptop). Progress is saved, so an interrupted build resumes."
    )
    if build_box.button(f"Build the {strategy} index", type="primary"):
        bar = build_box.progress(0.0, "Starting…")
        with timed("build", user=user, strategy=strategy, items=pending, model=model) as fields:
            pipeline.build(llm, lambda done, total: bar.progress(done / total, f"{done} / {total}"))
        if not fields.get("failed"):
            st.rerun()

if guide := st.session_state.get("info"):
    # The chat stays in session_state underneath, so going back loses nothing.
    choice = {
        "RAGS.md": strategy,
        "CHUNKING_STRATEGIES.md": chunking,
        "EMBEDDINGS.md": embedding,
        "RETRIEVERS.md": retriever,
        "EVALUATION_STRATEGIES.md": None,
    }[guide]
    info_page(guide, choice)
    st.stop()

for question, answered_by, result in st.session_state.history:
    st.chat_message("user", avatar=USER_ICON).write(question)
    with st.chat_message("assistant", avatar=BADGES[answered_by][0]):
        st.write(result.answer)
        answer_details(result, answered_by)

if not rag.sources():
    placeholder = "Upload and index a document first"
elif pending:
    placeholder = f"Build the {strategy} index first (sidebar)"
else:
    placeholder = "Ask about your documents"
# Pinned to the window bottom, with the evaluation strip fixed under it.
question = st.chat_input(placeholder, disabled=not rag.sources() or bool(pending))
# Emptied at the start of every run so, while a new answer is being generated, the previous
# answer's scores are not shown as if they were the new one's.
evaluation = st.bottom.empty()


def show_evaluation(result: Result | None, name: str, note: str = "") -> None:
    with evaluation.container():
        evaluation_strip(result, name, note)


if question:
    show_evaluation(None, strategy, f"{strategy} is answering — scoring follows.")
    st.chat_message("user", avatar=USER_ICON).write(question)
    with (
        st.chat_message("assistant", avatar=BADGES[strategy][0]),
        st.spinner(f"{strategy}: retrieving, answering, grading…"),
    ):
        # The question itself is not logged: it can hold what the documents are about.
        with timed(
            "question", user=user, strategy=strategy, retriever=retriever, model=model
        ) as fields:
            result = pipeline.ask(question, llm)
            grade = result.grade
            fields.update(
                chunks=len(result.chunks),
                relevance=grade and grade.relevance,
                grounded=grade and grade.groundedness,
            )
            st.write(result.answer)
            answer_details(result, strategy)
            st.session_state.history.append((question, strategy, result))

last = st.session_state.history[-1] if st.session_state.history else None
show_evaluation(last[2] if last else None, last[1] if last else strategy)
