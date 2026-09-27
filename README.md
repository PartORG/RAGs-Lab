# RAG Lab

A lab for building and comparing RAG strategies on local Ollama models. Nothing leaves the machine.
Several people can use one instance: each logs in and works on their own private documents.

Pick a strategy from the **RAG strategy** dropdown in the sidebar and ask the same question under
several of them. The numbers are the sections of [RAGS.md](src/docs/RAGS.md), the catalogue this lab
works through; each strategy lives in `src/rags/<folder>/rag.py`.

| # | Strategy | Folder | What it does here |
|---|---|---|---|
| 1 | Naive RAG | `naive_rag` | Top-4 cosine search, answer from those chunks. The baseline. |
| 2 | Retrieve-and-Rerank | `retrieve_and_rerank` | Top-20 search; a cross-encoder keeps the best 3. |
| 3 | Hybrid RAG | `hybrid_rag` | Dense + BM25 keyword search, fused with Reciprocal Rank Fusion. |
| 4 | Multi-Query / RAG-Fusion | `multi_query_rag` | 3 LLM rewrites of the question, each searched, fused. |
| 5 | HyDE | `hyde_rag` | Searches with an LLM-drafted "ideal answer" passage. |
| 6 | Conversational RAG | `conversational_rag` | Rewrites a follow-up into a standalone question first. |
| 7 | Hierarchical RAG | `hierarchical_rag` | Matches ~200-character children, reads their parent chunks. |
| 8 | RAPTOR | `raptor_rag` | Searches chunks plus a 2-level tree of cluster summaries. |
| 9 | Contextual RAG | `contextual_rag` | LLM-written header per chunk; then Hybrid + Rerank. |
| 10 | GraphRAG | `graph_rag` | LLM-extracted fact triples; walks 2 hops from the question's entities. |
| 11 | Multimodal RAG | `multimodal_rag` | Vision-model captions of PDF figures; answers looking at the figure. |
| 12 | Corrective RAG (CRAG) | `corrective_rag` | Grades each chunk; searches again or refuses rather than guess. |
| 13 | Self-RAG | `self_rag` | Checks: needs documents? chunks relevant? answer grounded? Loops, capped. |
| 14 | Adaptive RAG | `adaptive_rag` | Routes: no retrieval / Naive RAG / Multi-hop, by question complexity. |
| 15 | Iterative / Multi-hop RAG | `multi_hop_rag` | Up to 3 sub-question lookups, then one synthesized answer. |
| 16 | Speculative RAG | `speculative_rag` | A small model drafts per chunk; the big one picks the best draft. |
| 17 | Agentic RAG | `agentic_rag` | Search is a tool the model calls, in its own words, up to 4 steps. |
| 18 | Branched / Multi-Source RAG | `branched_rag` | Asks every index at once (chunks, keywords, summaries, figures). |
| 20 | Cache-Augmented Generation | `cag_rag` | No retrieval: the whole corpus goes in the prompt, if it fits. |
| 21 | Self-Query RAG | `self_query_rag` | Turns "page 40 of X" into exact file/page filters, then searches. |

**Not implemented, on purpose:** #19 Modular RAG is an engineering stance rather than a pipeline —
every strategy here is already a swappable module behind one `ask()` — and #22 Text-to-SQL needs a
database, while this lab holds uploaded documents.

Where this differs from the catalogue's code: every strategy sends 4 chunks to the model (3 for
the reranking ones), so they can be compared fairly; #12–14 and #17 are plain Python instead of
LangGraph and an agent framework; and **CRAG's web-search fallback is off by default**. Switched on
with its sidebar checkbox, only the rewritten question goes to DuckDuckGo, never your documents —
it is the one thing in this lab that leaves the machine.

## Requirements

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally:

  ```bash
  ollama pull nomic-embed-text  # embeddings
  ollama pull qwen3:8b          # chat; any installed chat model can be picked in the sidebar
  ollama pull qwen2.5vl:3b      # Multimodal RAG only
  ollama pull llama3.2:3b       # Speculative RAG's drafter; it falls back to the chat model
  ```

- Retrieve-and-Rerank and Contextual RAG download the `BAAI/bge-reranker-base` cross-encoder
  (~1.1 GB) from Hugging Face on first use (the Docker image has it built in). It runs on the
  CPU: ~3 s per question, ~2 GB of RAM. Torch is installed as its CPU-only build, since Ollama
  has the GPU; that keeps `.venv` around 1.6 GB instead of 6 GB.

**Memory:** on a 14 GB laptop, Ollama keeps the chat model's ~5–6 GB in RAM as well. Loading the
reranker or the vision model on top has been enough to make Ubuntu's out-of-memory guard close
the whole desktop session, so avoid running other heavy programs while using them.

## Run

```bash
uv sync
uv run main.py adduser ann      # create your first account (see Accounts below)
uv run main.py                  # then open http://localhost:8501 and log in
```

Upload PDF/TXT/MD files in the sidebar — several at once — and press **Save & index**. Every
strategy starts from those chunks, so one upload serves them all. Re-uploading a file with the
same name replaces it, and the **✕** beside a file deletes it after a confirmation: its chunks
leave every one of your indexes (all chunkings and embeddings), the file leaves the disk, and each
strategy drops whatever it had built from it (Multimodal RAG also deletes that PDF's extracted
figures).

Five strategies also build an index of their own, from a **Build** button that appears in the
sidebar when one is picked. It shows the estimated time; progress is saved as it goes, so an
interrupted build resumes, and after an upload only the new file is processed (RAPTOR rebuilds
its whole tree). Measured on this laptop with qwen3:8b for the 224-page, 591-chunk
*Code the Classics*:

| Strategy | Built from | Time |
|---|---|---|
| Hierarchical RAG | embeddings only, no LLM | under 1 min |
| RAPTOR | ~140 cluster summaries | ~9 min |
| Contextual RAG | one LLM call per chunk | ~12 min |
| GraphRAG | one LLM extraction per chunk | ~34 min |
| Multimodal RAG | one vision-model caption per figure (143 figures) | ~7 min |

### Accounts

There is no sign-up page: whoever runs the lab creates the accounts from a terminal.

```bash
uv run main.py adduser ann                          # locally
docker compose exec app python main.py adduser ann  # in Docker
```

It asks for the password twice and saves it as a salted scrypt hash. Rules:

- **Name:** 1–32 characters of `a-z`, `0-9`, `_` and `-` (it becomes a folder name).
- **Password:** at least 8 characters.
- **Forgotten password:** run `adduser` again with the same name; it sets a new password and
  keeps the account's documents.

Every user has their own uploads and indexes; nobody sees or deletes another user's. Accounts
and all indexes live in one SQLite file, `data/lab.db`; uploaded files in
`data/users/<name>/uploads/`. **Log out** is at the top of the sidebar. Streamlit keeps the login
per browser tab, so reloading the page asks for it again.

To delete an account with all its documents and indexes (it asks you to type the name again, and
there is no undo). A tab where that user is logged in is sent back to the login form on its next
click; the same happens after a password reset. An account later re-created under the same name
starts empty:

```bash
uv run main.py deluser ann                          # locally
docker compose exec -it app python main.py deluser ann  # in Docker
```

**Upgrading from before accounts:** the old `data/*.json` indexes and `data/uploads/` are no
longer read. Create an account, log in and upload the documents again; then the old files can be
deleted.

### Settings

All optional, read from the environment:

| Variable | Default | |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | where Ollama runs (read by the ollama client itself) |
| `RAG_CHAT_MODEL` | `qwen3:8b` | preselected chat model, and the one LLM-based chunkers use |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-base` | Hugging Face cross-encoder |
| `RAG_VISION_MODEL` | `qwen2.5vl:3b` | Multimodal RAG |
| `RAG_DRAFT_MODEL` | `llama3.2:3b` | Speculative RAG's drafter |
| `RAG_DATA` | `./data` | where `lab.db` and the uploads live |
| `RAG_LOG_LEVEL` | `INFO` | `DEBUG` also logs skipped PDF images |

### Logs

One line per event on stdout: logins, indexing, builds, deletes and questions, each with the
user, the choices involved and how long it took, e.g.

```
INFO rag_lab: question ok user='ann' strategy='Naive RAG' retriever='similarity' model='qwen3:8b' chunks=4 relevance=5 grounded=5 4.8s
```

A failure logs its traceback and shows the user a one-line error; the page keeps working. The
text of a question is never logged. Model output the lab could not use (a grade, a header, a
fact list) is logged as a warning, with the fallback taken.

### Docker

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec app python main.py adduser ann
```

The page is published on `127.0.0.1:8501` only: passwords travel over plain HTTP, so put an
HTTPS reverse proxy in front before letting others reach it. Ollama runs on the CPU unless
`nvidia-container-toolkit` is installed and the GPU block in `compose.yaml` is uncommented. The
reranker is baked into the image; the data and the Ollama models live on named volumes.

## Embeddings

The embedding model turns text into the vectors every strategy searches, so an index belongs to
the model that built it. **Embedding** in the sidebar picks from six — one folder each under
`src/embeddings/`, explained in [EMBEDDINGS.md](src/docs/EMBEDDINGS.md):

| Embedding | Dimensions | Where it runs | Index | Indexing | Peak RAM |
|---|---|---|---|---|---|
| nomic-embed-text *(default)* | 768 | Ollama | 8.9 MB | 6 s | — |
| mxbai-embed-large | 1024 | Ollama (`ollama pull`) | — | — | — |
| bge-small-en-v1.5 | 384 | this process, CPU | 6.6 MB | 21 s | 1.4 GB |
| bge-base-en-v1.5 | 768 | this process, CPU | 12.7 MB | 54 s | 2.0 GB |
| bge-m3 (multilingual) | 1024 | this process, CPU | 16.8 MB | 3 min 17 s | 3.2 GB |
| MiniLM multilingual | 384 | this process, CPU | 6.5 MB | 13 s | 1.8 GB |

The Ollama ones cost this app no memory; the Hugging Face ones load into it, which is worth
watching on a 14 GB machine where Ollama already holds 5–6 GB. Mind the input limits, too: MiniLM
multilingual reads only 128 tokens of a chunk and silently ignores the rest.

## Retrievers

The retriever decides which chunks come back for a question. **Retriever** in the sidebar picks
from six — one folder each under `src/retrievers/`, explained in [RETRIEVERS.md](src/docs/RETRIEVERS.md):

| Retriever | Ranks by | Extra cost |
|---|---|---|
| Similarity (top-k) *(default)* | cosine to the question | none |
| MMR (diverse) | cosine, penalised for repeating a pick | one extra embed per chunk |
| Similarity with a floor | cosine, dropping anything under 0.5 | none |
| BM25 (keywords) | term statistics, no embeddings | ~50 ms |
| Hybrid (dense + BM25) | both lists fused by rank | ~50 ms plus a second search |
| Reranked (cross-encoder) | a cross-encoder reading query and chunk together | ~3 s on the CPU |

It applies to every strategy that simply searches the index — Naive, Conversational, HyDE,
Multi-Query, Multi-hop, Corrective, Self-RAG, Speculative, Agentic, RAPTOR and Multimodal — while
the strategies built around their own retrieval (Hybrid, Retrieve-and-Rerank, Contextual,
Hierarchical, GraphRAG, Branched, Self-Query) keep theirs. Switching is free: a retriever writes
nothing to disk, so it needs no re-indexing.

## Chunking

Retrieval can only return a chunk, so where the cuts fall decides what any answer can be built
from. **Chunking** in the sidebar picks how, from eleven strategies — one folder each under
`src/chunking_strategies/`. [CHUNKING_STRATEGIES.md](src/docs/CHUNKING_STRATEGIES.md) explains what each
one is, what data suits it, and its pros and cons; the counts below are the same 224-page book
cut eleven ways:

| Chunking | Chunks | Indexing |
|---|---|---|
| Recursive characters *(default)* | 591 | instant |
| Fixed characters | 551 | instant |
| Whole sentences | 543 | instant |
| Sentence window | 4743 | instant |
| Paragraphs | 251 | instant |
| Document structure | 251 | instant |
| Token budget | 525 | ~10 s |
| Semantic breakpoints | ~700 | ~1.5 min |
| Model-chosen boundaries | ~1900 | ~3 min |
| Contextual headers | 591 | ~13 min |
| Propositions | ~5000 | ~41 min |

Each chunking keeps its **own indexes** per user and embedding (rows in `data/lab.db`), so
switching costs nothing and the work already built (a GraphRAG extraction, say) is never thrown
away. Pick one the
documents have not been indexed with yet and the sidebar offers to index them, with an estimate
first. The four slowest use an LLM at indexing time; they always use the default chat model, not
the one selected for chatting, so an index cannot change meaning because a dropdown moved. No
chunk exceeds 3000 characters, because `nomic-embed-text` reads about 2048 tokens and would
silently ignore the rest.

The page opens on a login form. Once logged in, it has three parts:

- **Left drawer:** who is logged in and **Log out**, the strategy, the Ollama model, the
  **Build** button when one is needed, and uploading and indexing documents.
- **Middle:** the chat, which is where the room goes. Under each answer sit the strategy's badge
  and two collapsed details — **What the strategy did** (rewrites, drafts, routing, verdicts,
  hops) and **Retrieved chunks** (what the model was actually given).
- **Bottom:** a static strip, 15% of the window, scoring the latest answer only. It never grows
  or moves, and keeps its place while an answer is being generated:

| In the strip | Meaning |
|---|---|
| Answers the question (1–5) | Does the answer address what was asked? |
| Grounded in the chunks (1–5) | Is every claim backed by what was retrieved? A refusal claims nothing, so it scores 5. |
| Chunks found | How many passages the model was given. |
| Best match | Cosine similarity between the question and the closest of them. |
| Judge | One line on why, from the model that answered. |

**ℹ️ buttons.** Each of the four dropdowns (RAG strategy, Embedding, Chunking, Retriever) has an
ℹ️ beside it. It replaces the chat with that choice's section of the guide in `src/docs/`: what it
is, its pros and cons, and for strategies a diagram of how it works. **← Back to chat** returns
with the conversation intact, and picking another option while reading shows that one instead.

The two 1–5 grades come from the same model grading its own answer: a self-check, not ground
truth. [EVALUATION_STRATEGIES.md](src/docs/EVALUATION_STRATEGIES.md) describes the other ways to
evaluate a RAG — retrieval metrics, claim-level faithfulness, reference answers, pairwise
comparison and more — with pros, cons and a diagram each; the ℹ️ beside *Answers are
self-graded* in the sidebar opens it in the app. Answers-the-question 1 with grounded 5 is the signature of "I don't know" — retrieval
found nothing.

Every strategy has its own icon and colour, shown on its answers and in the strip; blue and the
person icon always mean you. The colours were generated in OKLCH at even lightness and validated
for contrast, but 20 hues cannot all be told apart by a colour-blind reader, so the icon and the
name are always beside the colour and it never carries meaning alone.

## Layout

```
main.py                     starts the Streamlit page; `adduser` / `deluser NAME` manage accounts
Dockerfile, compose.yaml    the image, and the app + Ollama for `docker compose up`
src/frontend/app.py         the Streamlit page
src/rags/<strategy>/rag.py  one retrieval strategy each; naive_rag holds the shared answer + grade
src/chunking_strategies/    one chunking strategy each (see CHUNKING_STRATEGIES.md)
src/embeddings/             one embedding model each (see EMBEDDINGS.md)
src/retrievers/             one retriever each (see RETRIEVERS.md)
src/docs/                   the guides above, also shown by the ℹ️ buttons in the app
src/storage/                accounts and indexes, in SQLite
tests/                      uv run pytest (fake models, no Ollama needed)
```
