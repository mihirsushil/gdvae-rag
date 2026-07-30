# gdvae-rag — Deep Dive & Talking Points

A reference document for actually understanding (and being able to explain) every stage of this project. The README is for someone setting the project up; this is for you, to internalize the *why* behind each decision so you can defend it in a conversation.

---

## 1. The big picture

```
fetch → chunk → embed + index (dense & sparse) → hybrid search → re-rank → MCP tools
```

Every stage exists to solve one problem: **an LLM answering questions about a niche research codebase from memory will hallucinate.** This pipeline instead finds the actual relevant text and hands it to the model with a citation, so answers are grounded and verifiable.

Two families of retrieval combine at the middle: **dense** (embeddings — understands meaning) and **sparse** (BM25 — understands exact words). Neither alone is reliable; the interesting engineering is in how they're fused and refined.

---

## 2. Step 1 — Fetch (`ingest/fetch_sources.py`)

**What happens:** `git clone --depth 1` pulls the GD-VAE repo. `requests.get()` downloads 3 PDFs from arXiv (the GD-VAE paper itself, plus two background papers — the original VAE paper by Kingma & Welling, and a geometric deep learning survey by Bronstein et al.).

**Why background papers, not just the target repo:** the corpus needs to answer *prerequisite* questions too ("what's the reparameterization trick?"), not just questions about GD-VAE specifically — otherwise the system can retrieve GD-VAE code that *uses* a concept but never explains it.

**Tools:** `git` (subprocess), `requests` (HTTP).

---

## 3. Step 2 — Chunking (`ingest/chunk.py`)

**The problem chunking solves:** you can't embed or search a whole 400-page corpus as one blob — you need to break it into pieces small enough to be individually relevant, but large enough to keep context. The size and split points you choose directly determine retrieval quality; this is the single highest-leverage decision in a RAG system, and it's also the part most tutorial-following implementations skip by just doing naive fixed-size splitting everywhere.

**Three different strategies, chosen per content type:**

- **Python source → AST-based splitting.** Python's built-in `ast` module parses source code into a syntax tree (the same structure the Python interpreter itself builds before running your code). We walk the tree for top-level `FunctionDef`/`ClassDef` nodes and use `ast.get_source_segment()` to pull out the *exact original source text* of each one. This guarantees a function's signature, docstring, and full body always stay together as one chunk — never arbitrarily cut mid-function, which is what a naive line-count splitter would do.
- **Markdown/paper text → header-aware splitting.** A regex (`^#{1,3}\s+(.*)$`) finds section headers, and text is split at those boundaries — so a chunk corresponds to a coherent section, not a random slice. If a section is still too long (>400 words), it falls back to a sliding window (400 words, 60-word/~15% overlap) *within* that section. The overlap matters: it prevents a sentence that happens to fall right at a window boundary from losing its context in both neighboring chunks.
- **PDF text → per-page extraction, then windowing.** `pypdf.PdfReader` extracts text per page (this also gives us page numbers for free, which become part of the citation), then each page is split the same sliding-window way as long markdown sections.

**Output:** `data/chunks.jsonl` — one JSON object per chunk: `{id, source, doc_type, section, page, text}`. **385 chunks** total: 266 from the 3 PDFs, 86 from Python functions/classes, 33 from markdown sections.

**Tools:** `ast` (stdlib), `re` (stdlib), `pypdf`.

---

## 4. Step 3 — Embedding + Indexing (`ingest/build_index.py`)

This step builds the two things that actually get *searched*.

### Dense index (meaning-based)

An **embedding model** (`BAAI/bge-small-en-v1.5`, run locally via `sentence-transformers`) converts each chunk's text into a vector of 384 numbers. The model was trained so that texts with similar *meaning* end up as vectors that are close together in that 384-dimensional space — even if they don't share any words. This is what lets a query like "curved latent space" retrieve a chunk that says "manifold" without ever using the word "curved."

Once every chunk is a vector, `faiss` (a vector search library from Meta) stores them in an `IndexFlatIP` — "flat" means brute-force: to search, it computes the similarity between your query vector and *every single* stored vector and returns the closest ones. That sounds naive, but it's exact (no approximation error) and completely fine at this scale (385 vectors — comparing against all of them takes microseconds). Approximate nearest-neighbor indexes (like FAISS's `IVF` or `HNSW` variants) exist to make this fast at millions/billions of vectors by trading exactness for speed — unnecessary complexity here.

Vectors are **normalized** before indexing, which makes inner product equivalent to cosine similarity (the standard way to measure "how similar are these two directions," ignoring magnitude).

### Sparse index (keyword-based)

`rank_bm25`'s `BM25Okapi` builds a classic **term-frequency / inverse-document-frequency** ranking index. Conceptually: a chunk scores higher for a query term if that term appears often *in that chunk* (term frequency) but is rare *across the whole corpus* (inverse document frequency — common words like "the" contribute almost nothing). This is what plain full-text search engines are built on, and it's very good at exact/rare-term matches that embeddings can blur — a specific function name, an error code, a symbol.

**Tools:** `sentence-transformers` (runs the embedding model), `faiss` (stores/searches vectors), `rank_bm25` (keyword index), `numpy` (array plumbing).

---

## 5. Step 4 — Hybrid search (`retrieval/hybrid_search.py`)

When `search(query)` runs:

1. **Dense search** — embed the query, ask FAISS for the top 30 chunks by cosine similarity.
2. **Sparse search** — tokenize the query (lowercase, split on whitespace), ask BM25 for the top 30 by keyword score.
3. **Reciprocal Rank Fusion (RRF)** — merge the two ranked lists into one. Each chunk's fused score is `sum of 1/(60 + rank)` across every list it appears in (rank 0-indexed, `60` is a smoothing constant). A chunk ranked #1 in both lists scores far higher than one ranked #1 in only one list.

**Why RRF instead of just averaging or weighting the two raw scores:** cosine similarity scores and BM25 scores live on completely different, incompatible scales (cosine is bounded [-1, 1]; BM25 is an unbounded, corpus-dependent number). Combining them numerically would require normalizing both onto a comparable scale — fragile and dataset-dependent. RRF sidesteps the whole problem because it only cares about *rank position*, not the underlying score magnitude, so it needs zero tuning to work reasonably well.

**Tools:** same as step 3, used at query time instead of index-build time.

---

## 6. Step 5 — Re-ranking (`retrieval/rerank.py`)

The hybrid step returns 20 candidates — good recall (the right answer is *probably* in there), but not great precision (the ordering within those 20 isn't very trustworthy). Re-ranking fixes the ordering.

**Bi-encoder vs. cross-encoder — the key distinction:** the embedding model from step 3 is a **bi-encoder** — it encodes the query and each passage *independently* into vectors, then compares vectors. That's what makes it fast enough to search a whole corpus (you can pre-compute every chunk's vector once, offline). A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) instead feeds the query *and* a candidate passage into the model *together*, letting it directly attend between them and output one relevance score. This is much more accurate — but it requires a fresh forward pass for every single (query, passage) pair, so it's too slow to run against an entire corpus. The standard pattern (used here) is: cheap bi-encoder narrows a huge corpus down to a short list, expensive cross-encoder re-scores just that short list.

**Tools:** `sentence-transformers`'s `CrossEncoder`.

---

## 7. Step 6 — MCP server (`server.py`)

**What MCP is:** the Model Context Protocol is a standard way for an LLM client (like Claude Code or Claude Desktop) to discover and call external tools/data sources. A server declares tools; the client's model decides when to call them mid-conversation.

**`FastMCP`** (from the `mcp` Python SDK) is a thin wrapper: you write a normal Python function, decorate it with `@mcp.tool()`, and FastMCP auto-generates the tool's schema (name, description, parameters) directly from the function's **name, docstring, and type hints** — no separate schema-writing step.

**Three tools, deliberately kept thin:**
- `search(query, top_k)` — runs the full pipeline (steps 4+5) and formats results with citations (source file, section/function name, page number)
- `get_document(source)` — returns the full raw text of one source, for when the model wants more context than a chunk gives
- `list_sources()` — lists everything in the corpus

**Design choice — the server does *not* call an LLM itself to synthesize an answer.** It only retrieves and returns grounded text; the *calling* model (Claude) does the reasoning/synthesis. This is the correct MCP pattern (separation of concerns: the server owns data access, the client owns reasoning) and it also avoids needing to manage a separate API key/LLM call inside the server itself.

`mcp.run()` starts the server communicating over **stdio** (stdin/stdout) — the simplest MCP transport, appropriate for a locally-run tool (vs. an HTTP transport, which would matter for a server other people connect to remotely).

**Registration:** `claude mcp add gdvae-rag -s user -- <venv-python> server.py` writes an entry into `~/.claude.json` telling Claude Code to spawn this exact command as a subprocess at session start. `-s user` scope makes it available globally, not tied to one project directory.

**Tools:** `mcp` (the Python SDK).

---

## 8. Evaluation (`eval/`)

**Recall@5** is the metric: for each test question, is a chunk from the *correct* source document present somewhere in the top 5 results? This measures retrieval quality specifically — it doesn't test whether an LLM's final synthesized answer is good, just whether the system found the right raw material for it to work with. That's a deliberate scope choice: retrieval quality and generation quality are separable, and this project is about the retrieval half.

18 hand-written questions spanning code, repo docs, and all 3 papers → **16/18 = 88.9%**.

**The spot check is the more interesting result.** Querying the literal function name `map_clifford_torus` (an exact-term query, exactly BM25's supposed home turf) showed BM25 *alone* completely failing to find it. Why: BM25's tokenizer here is a naive `.lower().split()` on whitespace — it doesn't strip punctuation. So in the source line `def map_clifford_torus(input,params):`, the token is literally `map_clifford_torus(input,params):`, not `map_clifford_torus` — the query term never exact-matches anything. Dense search found it immediately, because embeddings work on meaning/subword structure, not exact string matching, so punctuation glued to an identifier doesn't break it. Hybrid fusion recovered the correct file in 3 of the top 5 slots.

This is worth remembering specifically because it's the *opposite* of the textbook pitch for hybrid search ("BM25 wins on exact terms, embeddings win on meaning") — here embeddings rescued a case where the sparse retriever's naive tokenizer broke on exact terms. That's a genuinely more interesting and defensible story than the textbook one, because it came from actually running the system, not from reciting theory.

---

## 9. Anticipated questions & how to answer them

**"Why not just use LangChain / LlamaIndex?"**
Because the goal was to actually understand and be able to explain every design decision — chunking strategy, fusion method, index type — rather than inherit a framework's defaults. A framework wrapper answers "what did you build," this answers "why does it work."

**"Why hybrid search instead of just embeddings?"**
Embeddings alone can miss exact terms (identifiers, rare technical vocabulary) that don't have strong semantic neighbors. BM25 alone misses paraphrases/synonyms. The corpus here mixes prose (papers, docs) and code (exact symbol names), so both failure modes are live risks — hybrid hedges against both. (And the spot check shows the *reverse* failure mode also happens in practice — worth mentioning, since it shows you tested the assumption instead of just asserting it.)

**"Why local embeddings instead of an API (OpenAI, Voyage, Cohere)?"**
No API key/cost, fully reproducible, and at ~400 chunks CPU inference is fast enough that GPU/API-scale throughput isn't needed. Trade-off: a paid API embedding model would likely have somewhat higher retrieval quality — a reasonable choice at larger scale or in production, not justified here.

**"Why AST-based chunking for code instead of fixed-size windows?"**
Fixed windows can and do split a function signature from its body, or a docstring from the code it documents — breaking retrievable units at arbitrary line counts. AST parsing chunks along the language's own semantic boundaries, so what gets embedded and retrieved is always a complete, meaningful unit.

**"How would this scale to a much bigger corpus?"**
Swap the flat FAISS index for an approximate one (IVF/HNSW) to keep search sub-linear, and BM25 would need a real inverted-index engine (rather than `rank_bm25`'s in-memory Python approach, which recomputes scores by scanning). The chunking/fusion/rerank architecture wouldn't need to change.

**"What's the actual weakness in this system?"**
Two honest ones: (1) BM25's tokenizer is naive — punctuation-adjacent identifiers can be missed, as the spot check showed; a proper tokenizer (e.g. splitting on `[\W_]+` or reusing the embedding model's subword tokenizer) would fix it. (2) Recall@5 only validates that the *retriever* found the right passage — it says nothing about whether the *final synthesized answer* (once Claude reads the retrieved chunks) is actually correct; that would need a separate answer-quality eval (e.g. an LLM-judge comparing generated answers to reference answers).

---

## 10. Full tool/library reference

| Tool | Stage | What it actually does |
|---|---|---|
| `git` | fetch | clones the source repo |
| `requests` | fetch | downloads PDFs over HTTP |
| `ast` (stdlib) | chunk | parses Python into a syntax tree; extracts function/class source segments |
| `re` (stdlib) | chunk | detects markdown headers via regex |
| `pypdf` | chunk | extracts text (and page numbers) from PDFs |
| `sentence-transformers` | index, search, rerank | runs the embedding model (bi-encoder) and the reranker (cross-encoder) |
| `faiss` | index, search | stores dense vectors; brute-force nearest-neighbor search |
| `rank_bm25` | index, search | keyword-frequency (BM25) ranking |
| `numpy` | index, search | array operations (argsort, dtype handling) between the above |
| `mcp` | serve | exposes Python functions as tools an LLM client can call, over stdio |
