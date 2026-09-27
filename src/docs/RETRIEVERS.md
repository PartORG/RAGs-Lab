# Retrievers

A retriever answers one question: *given this text, which chunks come back?* It sits between the
index and the strategy — after chunking and embedding have decided what exists, and before a
strategy decides what to do with it.

**It is a query-time choice, so it costs nothing to switch.** Unlike a chunking or an embedding,
a retriever writes nothing to disk and needs no re-indexing: pick another one and ask the same
question again. Implementations live in `src/retrievers/<folder>/retriever.py` and are chosen
from the **Retriever** dropdown.

**Who obeys it.** Every strategy that simply searches the index goes through one method,
`NaiveRAG.search`, so the choice reaches all of them at once: Naive, Conversational, HyDE,
Multi-Query, Multi-hop, Corrective, Self-RAG, Speculative, Agentic, RAPTOR (its leaves) and
Multimodal (its text side). The strategies whose *whole point* is their own retrieval keep it:
Hybrid RAG, Retrieve-and-Rerank, Contextual RAG, Hierarchical RAG, GraphRAG, Branched and
Self-Query. So "Hybrid" as a retriever and Hybrid RAG as a strategy are the same idea at two
layers: one makes every strategy hybrid, the other is hybrid on its own.

**Every retriever reports a cosine similarity**, even when it ranked by something else, so the
*best match* number in the bottom strip stays comparable across all six.

| # | Retriever | Folder | Ranks by | Extra cost per question |
|---|---|---|---|---|
| 1 | Similarity (top-k) *(default)* | `similarity` | cosine to the question | none |
| 2 | MMR (diverse) | `mmr` | cosine minus similarity to what is already picked | one extra embed per chunk |
| 3 | Similarity with a floor | `threshold` | cosine, then drops the weak | none |
| 4 | BM25 (keywords) | `bm25` | term statistics | ~50 ms to build the index |
| 5 | Hybrid (dense + BM25) | `hybrid` | both lists fused by rank | ~50 ms plus a second search |
| 6 | Reranked (cross-encoder) | `reranked` | a cross-encoder reading query and chunk together | ~3 s on the CPU |

---

## 1. Similarity (top-k) — *the default*

**What it is.** Embed the question, return the k nearest chunks by cosine. The whole of classic
dense retrieval, and what this project used everywhere before the dropdown existed.

**Best for** almost everything, as a starting point: it is the cheapest, the most predictable,
and the baseline the other five have to beat on your corpus.

**Pros**
- One embedding call and one scan; nothing else to go wrong.
- Finds paraphrases — the reason to use embeddings at all.
- Scores are directly interpretable as the similarity the UI shows.

**Cons**
- **Always returns k**, however bad the best match is, so a question the documents cannot answer
  still arrives with four confident-looking chunks.
- Near-duplicates: a repeated passage can take every slot.
- Blind to exact tokens — error codes, identifiers, rare names.

---

## 2. MMR (diverse)

**What it is.** Maximal Marginal Relevance. Fetch a wide candidate pool, then build the result one
chunk at a time, each pick scored for relevance *minus* similarity to the chunks already chosen.
A relevance/diversity dial (`lambda_mult`, 0.5 here) sets the trade.

**Best for** corpora that repeat themselves — documentation with boilerplate, minutes, contracts,
anything where the same paragraph appears in five places — and for broad questions where four
angles beat four copies of one angle.

**Pros**
- Kills near-duplicate results, so the context window holds four different things.
- Better recall on questions whose answer is spread over several sections.
- Costs no model: the same embeddings, selected differently.

**Cons**
- Trades away some relevance by construction; the second chunk is not the second best.
- Another knob to tune, and the right value is corpus-specific.
- Needs the vectors in memory (fine here, a real constraint on a hosted vector DB).

**In practice.** Fetches 5k candidates before selecting k, and re-scores the winners against the
question so the UI still shows a plain cosine.

---

## 3. Similarity with a floor

**What it is.** Ordinary top-k, then throw away everything below a cosine floor (0.5 here). It can
return three chunks, one, or none.

**Best for** corpora with gaps, and any setup where a wrong answer costs more than no answer —
support bots, compliance lookups. It pairs naturally with Corrective RAG, which is built to notice
that it has nothing to work with.

**Pros**
- A question the documents cannot answer arrives with no context, so the model says so instead of
  improvising from the four least-bad chunks.
- Makes the corpus's gaps visible instead of hiding them behind plausible text.
- Costs nothing over plain similarity.

**Cons**
- The floor is corpus- and model-specific: cosine scales differ per embedding model, so 0.5 means
  different things with nomic and with bge-small.
- Set too high, it starves good questions; too low, it does nothing.
- Cuts recall precisely when the answer is worded very differently from the question.

---

## 4. BM25 (keywords)

**What it is.** The classic lexical ranker: term frequency, inverse document frequency, length
normalisation. No embeddings anywhere in the path.

**Best for** exact tokens — error codes (`ERR_771`), part numbers, function names, people's names,
anything a user copies and pastes — and as a sanity check on whether the embedding is helping at
all.

**Pros**
- Never misses an exact match, which dense retrieval does surprisingly often.
- No embedding model at query time, so it is fast and cheap.
- Interpretable: a hit is a hit because of words you can point to.

**Cons**
- No notion of meaning: a question worded differently from the document finds nothing.
- Vulnerable to vocabulary mismatch, the exact problem embeddings were invented for.
- Rebuilt from the chunks on each question here (~50 ms for 600 chunks; cache it for a big corpus).

---

## 5. Hybrid (dense + BM25)

**What it is.** Run both searches and merge the two ranked lists with Reciprocal Rank Fusion,
which reads only positions, so the incomparable score scales never have to be calibrated.

**Best for** most production systems — it is the default in enterprise search for a reason. Mixed
corpora where questions are sometimes conceptual and sometimes a copied identifier.

**Pros**
- Covers both failure modes: paraphrase and exact token.
- RRF needs no tuning to start working.
- Deterministic and explainable; each hit comes from a list you can inspect.

**Cons**
- Two searches per question instead of one.
- Fusion weights (and how wide each side fetches) do need tuning to get the last few points.
- Still single-shot: no reasoning, no verification.

---

## 6. Reranked (cross-encoder)

**What it is.** Fetch 5k candidates by cosine, then score each with a cross-encoder that reads the
question and the chunk *together*, and keep the k it rates highest.

**Best for** precision-critical work, and for corpora full of near-misses where the difference
between the right chunk and a plausible one is a detail an embedding averages away.

**Pros**
- The largest quality jump of anything in this list, per the reranking literature and visible here.
- Fixes the ordering that a bi-encoder gets wrong, without touching the index.
- Composes with everything: any strategy that searches the index inherits it.

**Cons**
- ~3 s per question on the CPU with `bge-reranker-base`, against milliseconds for the rest.
- Loads a second model into this process (~2 GB with torch).
- Cannot recover a chunk the first, cheaper search never returned.

**In practice.** Uses the same cross-encoder as Retrieve-and-Rerank, loaded once and shared. The
similarity shown in the UI is still the cosine, not the cross-encoder's score, so numbers stay
comparable between retrievers.

---

## A measured example

The same question — *"How does the ball bounce off the bats in Boing?"* — asked of the same book,
same chunking, same embedding, same strategy (Naive RAG), changing only the retriever:

| Retriever | Answered? | Best match | Time |
|---|---|---|---|
| Similarity (top-k) | no — "I don't know" (1/5) | 0.68 | 5 s |
| **BM25 (keywords)** | **yes (5/5)** | 0.62 | 7 s |
| Hybrid (dense + BM25) | no (1/5) | 0.62 | 4 s |
| MMR (diverse) | no (1/5) | 0.68 | 4 s |
| Reranked (cross-encoder) | no (1/5) | 0.68 | 7 s |

The keyword ranker was the only one to find the page, because "bats" and "bounce" appear on it
verbatim while the embedding kept returning chunks that merely sound related — and note that the
winning retrieval had the *lower* cosine, which is exactly why "best match" is not a quality
score. Hybrid had that same chunk in one of its two lists and still lost it in the fusion, which
is what RRF's per-list weighting is for. One question is not a benchmark; it is a good reminder to
try two retrievers before believing either.

## Not implemented here, and why

- **Parent-document / small-to-big retrieval.** Match children, return parents. It needs a second
  store, so in this lab it is a strategy (Hierarchical RAG) rather than a retriever.
- **Metadata-filtered retrieval.** Restrict by file or page before searching. It is Self-Query RAG
  here, because the filter has to be parsed out of the question first.
- **Multi-query fusion.** Search several rewrites and merge. That is Multi-Query RAG — a strategy,
  because it needs an LLM call before it can retrieve at all.
- **Graph traversal.** GraphRAG, for the same reason: what it walks is not the chunk index.
- **Multi-vector / late interaction (ColBERT).** Would need a store holding one vector per token,
  which `InMemoryVectorStore` is not.

## Choosing one

- **Start with similarity.** It is the baseline everything else is measured against.
- **Answers repeat themselves?** MMR.
- **Would rather have no answer than a wrong one?** The floor, ideally with Corrective RAG.
- **Questions contain codes, names or identifiers?** BM25, or hybrid to keep paraphrase matching.
- **Want the best quality and can spare ~3 s?** Reranked.
- **One choice for general use?** Hybrid — it is what most production systems settle on.

Switching costs nothing, so the honest way to choose is to ask the same question under two
retrievers and compare *answers the question* and *grounded in the chunks* in the bottom strip.

## Adding one

Create `src/retrievers/<name>/retriever.py` with a function that takes the index (and
`get_reranker` if it needs a cross-encoder) and returns a `Retriever` — a callable of
`(question, k)` returning `(chunk, cosine)` pairs. Export it from that folder's `__init__.py` and
add one entry to `RETRIEVERS` in `src/retrievers/__init__.py`. The dropdown follows from the
registry, and `retrievers/common.py` has the `scored()` helper for attaching cosines when your
retriever ranks by something else.
