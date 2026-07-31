# gdvae-rag

An MCP server that answers questions about [GD-VAE](https://github.com/gd-vae/gd-vae) (Geometric Dynamic Variational Autoencoders) — a research codebase combining differential geometry, dynamical systems, and deep learning — grounded in the actual repo, paper, and background literature, with citations.

The retrieval pipeline (chunking, hybrid search, re-ranking) is built from scratch rather than wrapping a vector-DB SDK, so every stage is a deliberate, explainable design choice rather than a framework default.

## Why

Reading a research codebase and its paper cold is slow, and general-purpose LLMs will confidently hallucinate details about a niche, low-citation-count paper they've barely seen in training. This tool grounds every answer in an actual retrieved passage — code, docs, or paper text — with a citation you can go verify.

## Corpus

1. [`gd-vae/gd-vae`](https://github.com/gd-vae/gd-vae) — README, docs, examples, and the core `src/` Python package
2. GD-VAE paper — [arXiv:2206.05183](https://arxiv.org/abs/2206.05183)
3. Kingma & Welling, *Auto-Encoding Variational Bayes* — [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) (VAE fundamentals)
4. Bronstein et al., *Geometric Deep Learning* — [arXiv:2104.13478](https://arxiv.org/abs/2104.13478) (geometry background)

## Architecture

```
fetch → chunk → embed + index (dense & sparse) → hybrid search → re-rank → MCP tools
```

- **Chunking** is structure-aware, not naive fixed-size windows:
  - Python source is split with the `ast` module — one chunk per top-level function/class, so a function and its docstring never get split apart mid-body
  - Markdown/paper text is split on headers, falling back to a ~400-word sliding window (60-word overlap) for long sections
  - Every chunk carries metadata (source file, section/function name, PDF page number) — this is what powers citations
- **Retrieval is hybrid**: dense embeddings (`BAAI/bge-small-en-v1.5`, local, no API key) via a FAISS flat index, plus BM25 (`rank_bm25`) over the same chunks, fused with **Reciprocal Rank Fusion** — RRF sidesteps having to normalize two incompatible score scales
- **Re-ranking**: a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the top ~20 fused candidates for the final top-k
- **MCP server stays a thin retrieval layer** — it returns grounded, cited chunks; the calling model (Claude) does the reasoning. Three tools:
  - `search(query, top_k)` — hybrid search + rerank, returns cited passages
  - `get_document(source)` — full text of one source
  - `list_sources()` — everything in the corpus

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python3 ingest/fetch_sources.py   # clone repo + download papers
.venv/bin/python3 ingest/chunk.py            # -> data/chunks.jsonl
.venv/bin/python3 ingest/build_index.py      # -> data/index.faiss, data/bm25.pkl

claude mcp add gdvae-rag -s user -- "$(pwd)/.venv/bin/python3" "$(pwd)/server.py"
```

## Example

```
> search("what is the adjoint method used for in GD-VAE")

[1] gd-vae-repo/README.md § GD-VAEs: Geometric Dynamic Variational Autoencoders
    Adjoint Methods for General Latent Spaces: Provides approaches for
    handling general latent spaces through solving adjoint problems...

[2] gd-vae-repo/paper/paper.md § Example Package Usage
    ...By accommodating general topologies, the methods can help
    facilitate obtaining more parsimonious representations...
```

## Evaluation

18 hand-written questions spanning code (`src/*.py`), repo docs, and the three papers, checked against **Recall@5**: does a correctly-sourced chunk appear in the top 5 results?

**Recall@5: 15/18 (83.3%)** — run with `.venv/bin/python3 eval/run_eval.py`

The misses are informative rather than embarrassing: they're mostly paper/citation questions where a background-paper chunk that also legitimately answers the question isn't the one labeled "correct," or where a point-cloud-mapping chunk loses out to adjacent, related ones. Recall dipped slightly after fixing the tokenizer below — expected, since re-tokenizing the whole corpus reshuffles BM25's rankings everywhere, not just for the query it was fixed for. Worth the trade: exact-identifier lookups are a more important guarantee for a code-search tool than a couple points of recall on paraphrase-heavy questions.

**Spot check — why hybrid, not just one retriever:** querying the exact function name `map_clifford_torus`, BM25 alone now finds `src/geo_map.py` (tokenization was fixed to stop gluing trailing punctuation like `(input,params):` onto identifiers — see `retrieval/tokenize.py` and `tests/test_retrieval.py`), and dense embeddings find it immediately on semantic similarity regardless. Hybrid fusion puts the correct file in 3 of the top 5 slots. Before the fix, BM25 alone completely missed it — a real illustration of why relying on a single retrieval strategy, or trusting a naive tokenizer, is risky.

## Design notes

- **Local embeddings over an API** — free, reproducible, no key management, and the corpus (a few hundred chunks) is trivially small for CPU inference
- **Flat FAISS index over ANN** — at this scale, approximate search buys nothing but risk; exact search is fast enough
- **AST-based code chunking over fixed windows** — keeps a function's signature, body, and docstring as one retrievable unit instead of splitting it arbitrarily
- **RRF over score-weighted fusion** — BM25 and cosine-similarity scores live on different scales; RRF only needs rank order, which is more robust with no tuning
