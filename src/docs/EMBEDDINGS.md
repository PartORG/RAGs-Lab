# Embeddings

An embedding model turns text into a vector, and retrieval is nothing but comparing those
vectors. It sits underneath everything else in this lab: every strategy's search, the semantic
chunker's breakpoints, and the similarity numbers in the bottom strip. A chunking strategy decides
*what* can be retrieved; the embedding decides *whether it will be*.

**An index belongs to the model that built it.** The query has to be embedded by the same model,
in the same space, or the nearest neighbours mean nothing — so each embedding keeps its own index
files, and switching asks to index the documents again. Implementations live in
`src/embeddings/<folder>/embedder.py` and are chosen from the **Embedding** dropdown.

**Measured** on the 224-page, 591-chunk *Code the Classics* PDF with the default recursive
chunking, on a laptop with an RTX 4070 (8 GB) and 14 GB of RAM. "Peak RAM" is this app's own
process: the Ollama models run in Ollama and cost it nothing.

| # | Embedding | Folder | Dims | Input limit | Index | Indexing | Peak RAM |
|---|---|---|---|---|---|---|---|
| 1 | nomic-embed-text *(default)* | `nomic` | 768 | 2048 tok | 8.9 MB | 6 s | — |
| 2 | mxbai-embed-large | `mxbai` | 1024 | 512 tok | — | — | — |
| 3 | bge-small-en-v1.5 | `bge_small` | 384 | 512 tok | 6.6 MB | 21 s | 1.4 GB |
| 4 | bge-base-en-v1.5 | `bge_base` | 768 | 512 tok | 12.7 MB | 54 s | 2.0 GB |
| 5 | bge-m3 | `bge_m3` | 1024 | 8192 tok | 16.8 MB | 3 min 17 s | 3.2 GB |
| 6 | MiniLM multilingual | `minilm_multilingual` | 384 | **128 tok** | 6.5 MB | 13 s | 1.8 GB |

Row 2 is the one model not installed here, so its figures come from the model card rather than
from a run.

The index is JSON, so its size follows the number of digits each float is written with, not only
the dimensions: bge-base and nomic are both 768 dimensions, yet bge-base's index is 40% larger.

---

## 1. nomic-embed-text *(the default here)*

**What it is.** Nomic AI's 137M-parameter retrieval model, served by Ollama: 768 dimensions,
Apache-2.0, with open training data and code. Small enough (274 MB) to sit on the GPU beside the
chat model, and the reason this lab starts with it.

**Best for** general English documents, and for any setup where the app should stay light: Ollama
holds the weights, so switching to it costs this process nothing at all.

**Pros**
- Fastest indexing here by a wide margin — 591 chunks in 6 s, because it runs on the GPU.
- No memory in the app itself, which matters when a reranker or vision model is also loaded.
- Longer inputs than the BGE family (2048 tokens as Ollama runs it), so chunks are rarely cut.
- Matryoshka-trained: its 768 dimensions can be truncated to 512 or 256 with modest loss if the
  index size ever matters.

**Cons**
- English-centric; other languages are much weaker.
- Middling on current leaderboards next to newer models of the same size.
- Quality depends on Ollama's build and quantisation (F16 here), not just the original weights.

**In practice.** The model card asks for task prefixes — `search_document:` when indexing,
`search_query:` when searching — and Ollama's template is bare `{{ .Prompt }}`, so this lab sends
none. Adding them made no measurable difference on three questions against this book (the right
page ranked first either way), so the plain form stays.

---

## 2. mxbai-embed-large

**What it is.** Mixedbread AI's 335M-parameter model, 1024 dimensions, also served by Ollama
(670 MB). Trained with the AnglE objective; at release it matched much larger models on MTEB.

**Best for** wanting better English retrieval than nomic without loading torch into this app —
the natural second Ollama model to try.

**Pros**
- Stronger than nomic on English retrieval benchmarks.
- Still served by Ollama: no memory cost in this process.
- Supports Matryoshka truncation and binary quantisation, so the wider vectors need not cost more
  storage.

**Cons**
- 512-token inputs, a quarter of nomic's, so long chunks are truncated.
- 1024 dimensions means a bigger index and slower brute-force search.
- Not installed here: run `ollama pull mxbai-embed-large` first, and take the table's figures as
  the model card's rather than measured.

---

## 3. bge-small-en-v1.5

**What it is.** BAAI's smallest English retrieval model: 33M parameters, 384 dimensions, loaded
into this process from Hugging Face. The cheap end of the quality curve, and far from the worst.

**Best for** large corpora, quick experiments, and any moment when index size or search speed
matters more than the last few points of accuracy — for example when trying a chunking strategy
before committing to a slow one.

**Pros**
- The smallest index here: 6.6 MB against nomic's 8.9 MB for the same chunks.
- Fast search: 384-dimension dot products over a brute-force store add up.
- Genuinely good for its size — it holds its own against models ten times larger.

**Cons**
- Runs on the CPU in this process: 21 s to index the book, and ~1.4 GB of RAM including torch.
- 512-token inputs.
- English only.

**In practice.** Needs the query instruction `"Represent this sentence for searching relevant
passages: "` on the query side only; `src/embeddings/bge_small/` carries it, and a test checks it
is not accidentally prepended to documents too.

---

## 4. bge-base-en-v1.5

**What it is.** The middle of the same family: 109M parameters, 768 dimensions — the same width as
nomic-embed-text, which makes it the like-for-like comparison when you want to know whether the
embedding or something else is the weak link.

**Best for** English corpora where retrieval quality is the priority and the indexing wait is
acceptable; a solid default if you would rather not depend on Ollama for embeddings.

**Pros**
- Stronger than bge-small at the same index width as the default.
- Same well-understood family, same instruction, same 512-token limit — an easy A/B.
- Fully local to the process: no Ollama round-trip per batch.

**Cons**
- 54 s to index the book against nomic's 6 s, all on the CPU, and 2.0 GB of memory here.
- 512-token inputs, so a 3000-character chunk is cut short.
- English only, and the vectors are twice the width of bge-small for a modest gain.

---

## 5. bge-m3

**What it is.** BAAI's multilingual flagship: 568M parameters, 1024 dimensions, 8192-token
inputs, 100+ languages. Unusually, one model produces three kinds of representation — dense,
sparse (lexical weights) and ColBERT-style multi-vector. This lab uses the dense output only.

**Best for** documents that are not in English, mixed-language corpora, and long chunks — whole
PDF pages or document-structure chunks, which the 512-token models truncate.

**Pros**
- The strongest retrieval quality available locally here, and by a clear margin on multilingual
  text.
- 8192-token inputs: nothing this lab produces comes close to being truncated.
- No query instruction to remember.

**Cons**
- Heavy: 3.2 GB of measured peak memory in this process, on top of the 5–6 GB Ollama holds. On a
  14 GB machine that is the combination that has previously triggered the out-of-memory killer.
- The slowest to index by a wide margin: 3 min 17 s against nomic's 6 s, all of it on the CPU.
- 1024 dimensions and the largest index of the six, at 16.8 MB.

**In practice.** Its sparse and multi-vector outputs are exactly what Hybrid RAG and a ColBERT
store would want; wiring those up would mean a store that can hold more than one vector per chunk.

---

## 6. MiniLM multilingual

**What it is.** `paraphrase-multilingual-MiniLM-L12-v2`: 118M parameters, 384 dimensions, 50+
languages, distilled from a larger multilingual teacher.

**Best for** non-English documents on a machine that cannot spare the memory for bge-m3 — and for
short texts specifically.

**Pros**
- Multilingual at 1.8 GB against bge-m3's 3.2 GB.
- The fastest of the local models here — 13 s, though partly because it reads only the first 128
  tokens of each chunk and simply never embeds the rest.
- Small vectors and a small index (6.5 MB).
- No query instruction needed.

**Cons**
- **A 128-token input limit** — the lowest here by a long way. A default 800-character chunk is
  roughly 200 tokens, so **about half of every chunk is silently ignored**. Pair it with a small
  chunk size (sentence window, or a few hundred characters) or it will quietly underperform.
- Trained for paraphrase similarity rather than question-to-passage retrieval, which is a
  different task: symmetric, not asymmetric.
- Weaker than bge-m3 wherever the memory is available for the latter.

---

## The wider landscape

Everything above is a dense single-vector model, which is what this lab's store expects. The
[MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is where the current ranking
actually lives — treat any list, including this one, as a snapshot.

**Local, served by Ollama.** `nomic-embed-text`, `mxbai-embed-large`, `bge-m3`, `all-minilm`,
`snowflake-arctic-embed2`, `granite-embedding`, `embeddinggemma`, `qwen3-embedding` (0.6B/4B/8B).
One `ollama pull` plus a folder here is all any of them needs.

**Local, from Hugging Face** (`sentence-transformers`, already a dependency): the **BGE** family,
**E5** (`intfloat/e5-*`, `multilingual-e5-*`), **GTE** (`Alibaba-NLP/gte-*`), **Jina v3** (long
context, task-specific adapters), **Stella**, **Qwen3-Embedding**, and **static** models
(`model2vec`) that embed thousands of chunks a second on a CPU at a real cost in quality.

**Hosted APIs** — none used here, since nothing in this lab leaves the machine: OpenAI
`text-embedding-3-small/large`, Cohere `embed-v4`, Voyage `voyage-3` and its code and legal
variants, Google `gemini-embedding`, Mistral, Jina, and Nomic's own API. Typically a little ahead
of the best open weights, priced per million tokens.

**Beyond one dense vector per chunk:**

- **Sparse and learned-sparse.** BM25 — hand-rolled here and used by Hybrid RAG, CRAG's fallback
  and Contextual RAG — plus **SPLADE**, which expands a query into weighted terms. Exact tokens
  (error codes, part numbers, function names) are where dense vectors are weakest, which is the
  whole argument for hybrid search.
- **Multi-vector / late interaction.** **ColBERT** keeps a vector per token and scores query
  tokens against document tokens; **ColPali** does it over page images, skipping text extraction
  entirely. Much stronger, much bigger indexes, and they need a store built for them.
- **Cross-encoder rerankers.** Not embeddings — no vectors, no index — but the same job at the
  other end of the pipeline, reading query and chunk together. This lab uses `bge-reranker-base`
  in Retrieve-and-Rerank and Contextual RAG.
- **Multimodal.** CLIP and SigLIP put images and text in one space, so a picture can be retrieved
  by a text query. Multimodal RAG here takes the other road: a vision model writes captions, and
  the captions are embedded as text.

## Things worth knowing

- **Dimensions are not quality.** 384 good dimensions beat 1024 mediocre ones. Width sets index
  size and search cost; training sets accuracy. Matryoshka-trained models (nomic, OpenAI v3,
  mxbai) can be truncated to a smaller width with only a small loss.
- **Query prefixes are not optional.** BGE and E5 are trained with an instruction on the query
  side, and omitting it quietly costs accuracy. Each folder here carries what its model expects;
  bge-m3 and the MiniLM models want none.
- **Input limits truncate silently.** 128 tokens for MiniLM multilingual, 512 for the BGE English
  models, 2048 for nomic as Ollama runs it, 8192 for bge-m3. Nothing warns you — the tail of the
  chunk simply never reaches the vector. It is the main reason no chunk here exceeds 3000
  characters.
- **Asymmetric vs symmetric.** Retrieval models are trained for short question against long
  passage; paraphrase models for two texts of similar length. Using a paraphrase model for
  retrieval works, but not as well.
- **Similarity numbers do not compare across models.** The same question on the same book scores
  0.65 best match with nomic and 0.74 with bge-small. Compare *answers the question* and *grounded
  in the chunks* between embeddings, not the cosine.
- **Switching always costs a re-index.** There is no migrating an index from one model to another.

## Choosing one

- **English documents, and you want the app to stay light:** nomic-embed-text. Nothing else here
  indexes in 6 s, and the weights live in Ollama rather than in this process.
- **A big corpus, or trying chunkings before committing:** bge-small — the smallest index, and
  good for its size.
- **You want to know whether the embedding is the weak link:** bge-base, the same 768 dimensions
  as the default, so only the model changes.
- **Documents not in English:** bge-m3 if 3.2 GB of memory is free, MiniLM multilingual if not —
  and with MiniLM, cut smaller chunks, or most of each one is never read.
- **Better English quality with no torch in this process:** `ollama pull mxbai-embed-large`.
- **Long chunks (document-structure or page chunking):** bge-m3 is the only one here that reads
  more than 2048 tokens; the BGE English models stop at 512.

Ask the same question under two embeddings and compare *answers the question* and *grounded in
the chunks* in the bottom strip — not the similarity number, which is not comparable between
models.

## Adding one

Create `src/embeddings/<name>/embedder.py` with a function returning a langchain `Embeddings` —
for a Hugging Face model, `LocalEmbeddings("org/model", query_prefix)` from `common.py` is the
whole implementation — export it from that folder's `__init__.py`, and add one entry to
`EMBEDDINGS` in `src/embeddings/__init__.py`. The dropdown, the per-embedding index files and the
re-index prompt all follow from the registry.
