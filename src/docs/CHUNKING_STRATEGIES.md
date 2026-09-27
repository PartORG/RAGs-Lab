# Chunking Strategies

Retrieval can only ever hand the model a chunk. Where the cuts fall therefore decides what any
answer can be built from — before a single retrieval strategy in [RAGS.md](./RAGS.md) gets a say.
A chunk that is too big drowns the answer in noise and wastes the context window; too small and
the sentence that mattered has lost the subject it was about.

The eleven strategies below are implemented in `src/chunking_strategies/<folder>/chunker.py` and
chosen from the **Chunking** dropdown in the sidebar. Each keeps its own indexes, so switching
between them throws nothing away and the same question can be asked of the same documents cut
several ways.

**Timings** are for the 224-page, 591-chunk *Code the Classics* PDF, measured on a laptop with an
RTX 4070 (8 GB), `nomic-embed-text` for embeddings and `qwen3:8b` where a chunker needs an LLM.
Embedding the chunks afterwards takes about 6 s on top.

| # | Strategy | Folder | Chunks | Median size | Indexing |
|---|---|---|---|---|---|
| 1 | Fixed characters | `fixed_size` | 551 | 800 | instant |
| 2 | Recursive characters *(default)* | `recursive` | 591 | 748 | instant |
| 3 | Whole sentences | `sentences` | 543 | 676 | instant |
| 4 | Sentence window | `sentence_window` | 4743 | 24 | instant |
| 5 | Paragraphs | `paragraphs` | 251 | 1497 | instant |
| 6 | Document structure | `document_structure` | 251 | 1498 | instant |
| 7 | Token budget | `token_based` | 525 | 713 | ~10 s |
| 8 | Semantic breakpoints | `semantic` | ~700 | — | ~1.5 min |
| 9 | Model-chosen boundaries | `llm_boundaries` | ~1900 | — | ~3 min |
| 10 | Contextual headers | `contextual_headers` | 591 | 900 | ~13 min |
| 11 | Propositions | `propositional` | ~5000 | — | ~41 min |

Counts marked ~ are extrapolated from a five-page sample rather than a full run.

---

## 1. Fixed characters

**What it is.** Cut every N characters, wherever that lands — mid-word, mid-sentence, mid-table.
No knowledge of the text at all. The baseline every other strategy has to beat.

**Best for** uniform machine-generated text with no structure worth respecting: logs, sensor
dumps, OCR sludge. Also the honest control when you want to know whether a cleverer chunker is
actually earning its cost.

**Pros**
- Nothing to tune, nothing to break, no dependencies.
- Perfectly predictable chunk count and size, so cost and latency are predictable.
- The fastest possible indexing path.

**Cons**
- Cuts sentences and words in half; the halves embed as noise.
- A fact split across the boundary is retrievable from neither side.
- Tables and code are mangled.

**In practice.** Add overlap (10–20%) to soften the boundary damage: the same text appears at the
end of one chunk and the start of the next, so a split fact survives in one of them. This
implementation defaults to no overlap precisely so the failure mode is visible.

---

## 2. Recursive characters *(the default here)*

**What it is.** The same size budget, but the cut is made at the most natural boundary that fits:
try paragraph breaks first, then line breaks, then sentence ends, then spaces. LangChain's
`RecursiveCharacterTextSplitter`, and the industry default for prose.

**Best for** general prose: manuals, books, articles, wikis, mixed PDFs. When in doubt, this one.

**Pros**
- Nearly always keeps sentences whole, at no cost over fixed-size.
- One knob (size) with a well-understood effect; 500–1000 characters suits most embedding models.
- Degrades gracefully: with no paragraph or sentence breaks it simply becomes fixed-size.

**Cons**
- Still blind to meaning: a section about two subjects is cut on length, not on topic.
- Overlap duplicates text in the index, inflating it and sometimes returning near-duplicate hits.
- Chunk boundaries move when a document is edited, so the whole file must be re-indexed.

**In practice.** 800 characters with 100 overlap here, applied per PDF page so no chunk spans two
pages and every chunk can cite its page.

---

## 3. Whole sentences

**What it is.** Split into sentences, then pack them until the next one would not fit. A sentence
is never cut in half.

**Best for** dense factual prose where a half-sentence is worse than a short chunk: legal text,
medical notes, specifications, anything with numbers and conditions.

**Pros**
- Every chunk starts and ends on a complete thought, so embeddings are cleaner.
- Chunks stay close to the target size without a hard cut.
- No model and no dependency: a regex on sentence punctuation.

**Cons**
- Sentence detection by punctuation misfires on abbreviations ("Fig. 3", "e.g."), code, and
  languages that punctuate differently.
- A single very long sentence, or text with no punctuation at all, produces an oversized chunk.
- Still no notion of topic.

**In practice.** Chunks here reached 1992 characters on code listings, which have few full stops.
An `nltk` or `spaCy` sentencizer handles abbreviations far better if that matters for your corpus.

---

## 4. Sentence window

**What it is.** One chunk per sentence for matching, stored together with the sentences either
side of it (two by default). The embedding is about a single thought; the text handed to the model
still carries its context. Also called small-to-big at sentence scale.

**Best for** question answering over long prose where the answer is one sentence, and for corpora
where precision matters more than index size: FAQs, interviews, transcripts.

**Pros**
- Very sharp matching: short text embeds one idea rather than an average of five.
- The model still sees the surrounding sentences, so a pronoun or a follow-on clause resolves.
- No LLM, no extra store.

**Cons**
- The index explodes: one chunk per sentence, heavily overlapping — 4743 for this book.
- Retrieval returns near-duplicate neighbours unless deduplicated.
- More embedding calls at index time, and a bigger index file.

**In practice.** Beware PDFs with hard line breaks: the sentence regex treats them as breaks, so
the median chunk here is only 24 characters — the windows are built from fragments, not sentences.
A proper sentencizer fixes this. The pattern LlamaIndex ships as `SentenceWindowNodeParser`. Hierarchical RAG
(strategy #7 in RAGS.md) is the same idea one level up: match children, read parents.

---

## 5. Paragraphs

**What it is.** Split on blank lines — the author's own unit of thought — and merge short
neighbouring paragraphs up to the size budget. A paragraph is never split unless it alone exceeds
the hard cap.

**Best for** well-formatted text where paragraphing is meaningful: Markdown, wikis, documentation,
email, anything written by a person rather than extracted from a layout.

**Pros**
- Boundaries the author intended, at zero cost.
- Chunk size adapts to the writing instead of forcing a target.
- Very legible chunks when read back in the retrieved-chunks table.

**Cons**
- Useless where blank lines are absent or meaningless: many PDF extractions, single-column OCR.
  On this book it produced exactly what document structure did, one chunk per page.
- Chunk sizes vary widely, which makes cost and context budgeting less predictable.
- A long paragraph still has to be cut somewhere.

---

## 6. Document structure

**What it is.** Follow the divisions the document already has: a Markdown section under its
heading, or a PDF page. Headings stay inside the chunk, so the embedding sees "Chapter 3 —
Collision detection" along with the body.

**Best for** structured documents: Markdown docs, HTML exports, standards, contracts, and PDFs
whose pages are self-contained (slide decks, books, forms).

**Pros**
- Chunks correspond to something a human would point at, which makes citations meaningful.
- Headings are strong retrieval signal, and keeping them inside costs nothing.
- Few, large chunks: less index, fewer near-duplicates.

**Cons**
- Chunks are large and uneven (1498 characters median here, up to the 3000 cap), so more of the
  context window goes on each hit.
- A section covering several topics is never split.
- Needs the structure to exist: a text dump with no headings becomes one chunk per file.

**In practice.** Splitters exist per format — `MarkdownHeaderTextSplitter` (used here),
`HTMLHeaderTextSplitter`, and language-aware splitters that cut code at function boundaries.

---

## 7. Token budget

**What it is.** Pack whole sentences until a real tokenizer says the budget is spent. Characters
are only a proxy for what the model actually counts.

**Best for** corpora where the character-to-token ratio is unstable: code, non-Latin scripts
(Chinese, Japanese, Arabic), heavy punctuation, mixed-language documents. Also whenever you are
close to an embedding model's input limit and need to be exact.

**Pros**
- Never silently overflows the embedding model's window.
- Comparable sizes across languages, unlike a character count.
- Cost per chunk is the number you are actually billed or throttled on.

**Cons**
- Needs a tokenizer, and ideally the *same* one the embedding model uses — rarely available.
- Slower than counting characters; the tokenizer is another download.
- Otherwise behaves exactly like sentence packing, so the gain is limited for English prose.

**In practice.** 200 tokens here, counted with the `BAAI/bge-reranker-base` tokenizer the lab
already downloads for reranking. That is a close relative of, not identical to, the tokenizer
`nomic-embed-text` uses.

---

## 8. Semantic breakpoints

**What it is.** Embed every sentence, compare each neighbouring pair, and cut where similarity
drops — the lowest 30% of boundaries here. Chunk edges land where the subject changes rather than
where a counter runs out.

**Best for** unstructured prose that wanders between topics without headings: transcripts,
meeting notes, interviews, long reports, scraped pages.

**Pros**
- Chunks are about one thing, which is exactly what an embedding represents best.
- Adapts to the text: a tight technical passage stays whole, a rambling one is cut often.
- Needs no LLM — only the embedding model already in use.

**Cons**
- One embedding call per sentence at index time (~1.5 min for this book; far more on a big corpus).
- The percentile threshold is a knob with no obvious right value, and it is corpus-dependent.
- Similarity between adjacent sentences is a weak proxy for "new topic": lists and dialogue fool it.

**In practice.** Popularised by LlamaIndex's `SemanticSplitterNodeParser` and LangChain's
`SemanticChunker`. Percentile thresholding is the usual variant; standard-deviation and
interquartile variants exist.

---

## 9. Model-chosen boundaries (agentic chunking)

**What it is.** Number the sentences of a page, hand them to an LLM, and ask which ones start a
new topic. The model decides the cuts.

**Best for** documents whose structure is real but invisible to rules: mixed-format PDFs,
transcripts with topic shifts, pages combining narrative with tables or code.

**Pros**
- Understands the text rather than its punctuation, so boundaries match how a reader would split it.
- Handles formats no rule anticipated, including tables and code blocks.
- The prompt can be steered ("keep procedures whole", "never split a table").

**Cons**
- One LLM call per page (~3 min for this book), and the cost scales with the corpus.
- Non-deterministic: the same page may be split differently after a model upgrade.
- A model that answers badly silently returns one chunk per page, or nonsense boundaries.

**In practice.** Cheap here because the unit is a page. Asking per document, or letting the model
rewrite the text as it splits, gets expensive fast.

---

## 10. Contextual headers

**What it is.** Cut recursively, then prefix every chunk with one LLM-written sentence saying
where it sits: *"From the Boing chapter, explaining how the ball bounces off the bats."* The header
is embedded and indexed along with the chunk. Anthropic's contextual retrieval, applied at
chunking time so every strategy inherits it.

**Best for** corpora where chunks are meaningless alone — "the company grew 3%" — such as
financial filings, long technical manuals, and anything with heavy pronoun use across sections.

**Pros**
- Anthropic measured ~49% fewer retrieval failures with hybrid search, ~67% with reranking on top.
- Fixes the deepest flaw of chunking — lost context — at the source rather than at query time.
- Query path stays simple: it is just a chunk with a better first line.

**Cons**
- One LLM call per chunk (~13 min for this book), and re-chunking means re-enriching.
- The whole document must fit the enrichment model's context, or a substitute prefix must be used
  (the first 8000 characters, here).
- A wrong header actively misleads retrieval.

**In practice.** Contextual RAG (strategy #9 in RAGS.md) does the same thing in its own index and
adds hybrid search plus reranking on the query side. Use this chunker when you want *every*
strategy to benefit; use the strategy when you want to compare it against the others.

---

## 11. Propositions

**What it is.** The LLM rewrites each page as a list of standalone facts — "It was released in
1971" becomes "Computer Space was released in 1971" — and each fact becomes a chunk. Sometimes
called propositional or atomic chunking; see the *Dense X Retrieval* paper.

**Best for** fact-dense reference material that gets queried with precise questions: product
specifications, policies, knowledge bases, FAQs.

**Pros**
- Each chunk is self-contained by construction, so no chunk depends on its neighbours.
- Very sharp retrieval for one-fact questions, and easy to deduplicate across documents.
- Rewriting resolves pronouns, dates and subjects that plain chunking leaves dangling.

**Cons**
- The most expensive chunker here (~41 min for this book) and it rewrites text, so anything the
  model drops is simply gone from the index.
- Produces a very large number of tiny chunks (~5000), which raises embedding cost and makes
  whole-document questions harder, not easier.
- Style, tables and worked examples do not survive as facts; the retrieved text is no longer the
  author's.

**In practice.** Best combined with a strategy that can gather many small hits — Multi-Query,
RAG-Fusion or Multi-hop — rather than plain top-4 retrieval.

---

## Not implemented here, and why

- **Parent-child / hierarchical chunking.** Match small children, feed the model the large parent.
  It needs two stores rather than one list of chunks, so in this lab it is a retrieval strategy
  (#7 Hierarchical RAG in RAGS.md), not a chunker.
- **Late chunking.** Embed the whole document with a long-context embedding model, then pool the
  token vectors per chunk, so every chunk's vector knows the whole document. It needs token-level
  embeddings, which Ollama's embedding endpoint does not expose — it returns one pooled vector per
  input — and the query would have to be embedded by the same model.
- **Multimodal chunking.** Figures, tables and scanned pages as their own chunks. Implemented as a
  retrieval strategy (#11 Multimodal RAG), where a vision model captions each figure.

## Choosing one

- **Start with recursive characters.** It is the default for good reason.
- **Documents with real headings?** Try document structure; the headings are free retrieval signal.
- **Answers are single sentences?** Sentence window.
- **Chunks that make no sense alone?** Contextual headers, if you can afford the indexing.
- **Text that wanders between topics with no headings?** Semantic breakpoints.
- **Precise questions over reference data?** Propositions, paired with a multi-query strategy.
- **Not English, or lots of code?** Token budget, so the sizes mean something.

Ask the same question under two chunkings and compare the bottom strip: *answers the question*,
*grounded in the chunks*, *chunks found* and *best match* are all directly comparable, because
only the cutting changed.
