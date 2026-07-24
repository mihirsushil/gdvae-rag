"""Build the dense (FAISS) and sparse (BM25) indexes over data/chunks.jsonl."""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"
FAISS_PATH = REPO_ROOT / "data" / "index.faiss"
BM25_PATH = REPO_ROOT / "data" / "bm25.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_dense_index(chunks: list[dict]) -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"wrote dense index ({index.ntotal} vectors, dim={embeddings.shape[1]}) to {FAISS_PATH}")


def build_bm25_index(chunks: list[dict]) -> None:
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with BM25_PATH.open("wb") as f:
        pickle.dump(bm25, f)
    print(f"wrote BM25 index ({len(tokenized_corpus)} docs) to {BM25_PATH}")


def main() -> None:
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("no chunks found — run ingest/chunk.py first")
    build_dense_index(chunks)
    build_bm25_index(chunks)


if __name__ == "__main__":
    main()
