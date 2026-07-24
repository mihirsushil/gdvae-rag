"""Evaluate retrieval quality: Recall@5 over a hand-written question set."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.hybrid_search import dense_search, hybrid_search, sparse_search  # noqa: E402
from retrieval.rerank import rerank  # noqa: E402

QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
TOP_K = 5


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open() as f:
        return [json.loads(line) for line in f]


def source_matches(source: str, expected_sources: list[str]) -> bool:
    return any(expected in source for expected in expected_sources)


def evaluate() -> float:
    questions = load_questions()
    hits = 0
    rows = []
    for q in questions:
        candidates = hybrid_search(q["question"])
        top = rerank(q["question"], candidates, top_k=TOP_K)
        found = any(source_matches(c["source"], q["expected_sources"]) for c in top)
        hits += found
        rows.append((q["question"], found, [c["source"] for c in top]))

    recall = hits / len(questions)
    print(f"Recall@{TOP_K}: {hits}/{len(questions)} = {recall:.1%}\n")
    for question, found, sources in rows:
        mark = "PASS" if found else "FAIL"
        print(f"[{mark}] {question}")
        if not found:
            print(f"       retrieved: {sources}")
    return recall


def spot_check_hybrid_vs_dense(query: str) -> None:
    """Show a query where BM25's exact-term matching helps over pure dense search."""
    dense_only = dense_search(query, top_n=5)
    sparse_only = sparse_search(query, top_n=5)
    chunks = hybrid_search(query, top_n=5)

    print(f"\nSpot check — query: {query!r}")
    print("dense-only top sources:", [_chunk_source(i) for i in dense_only])
    print("bm25-only top sources: ", [_chunk_source(i) for i in sparse_only])
    print("hybrid fused sources:  ", [c["source"] for c in chunks])


def _chunk_source(idx: int) -> str:
    from retrieval.hybrid_search import _load_chunks

    return _load_chunks()[idx]["source"]


if __name__ == "__main__":
    evaluate()
    spot_check_hybrid_vs_dense("map_clifford_torus")
