"""Hybrid retrieval: BM25 + dense embeddings fused with Reciprocal Rank Fusion."""

import json
import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.tokenize import tokenize as _tokenize

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"
FAISS_PATH = REPO_ROOT / "data" / "index.faiss"
BM25_PATH = REPO_ROOT / "data" / "bm25.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

RRF_K = 60
FUSED_CANDIDATES = 20
DENSE_TOP_N = 30
SPARSE_TOP_N = 30


@lru_cache(maxsize=1)
def _load_chunks() -> list[dict]:
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=1)
def _load_dense_index():
    return faiss.read_index(str(FAISS_PATH))


@lru_cache(maxsize=1)
def _load_bm25():
    with BM25_PATH.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def dense_search(query: str, top_n: int = DENSE_TOP_N) -> list[int]:
    model = _load_embedder()
    index = _load_dense_index()
    query_vec = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")
    _, indices = index.search(query_vec, top_n)
    return [int(i) for i in indices[0] if i != -1]


def sparse_search(query: str, top_n: int = SPARSE_TOP_N) -> list[int]:
    bm25 = _load_bm25()
    scores = bm25.get_scores(_tokenize(query))
    ranked = np.argsort(scores)[::-1][:top_n]
    return [int(i) for i in ranked]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked lists without needing to normalize incompatible score scales."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def hybrid_search(query: str, top_n: int = FUSED_CANDIDATES) -> list[dict]:
    """Return the top_n fused candidate chunks (pre-rerank) for a query."""
    dense_ranked = dense_search(query)
    sparse_ranked = sparse_search(query)
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])[:top_n]
    chunks = _load_chunks()
    return [chunks[i] for i in fused]
