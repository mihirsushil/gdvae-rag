"""Score retrievers on the held-out set: does the returned text contain the gold span?

Usage: python eval/run_heldout.py {gdvae-rag|stair-zeroshot|stair-base|stair-finetuned}

Every system is scored by the same rule (eval/gold.py), so a chunk retriever and a section retriever
are comparable. Because sections are much longer than chunks, the words each system returns are
reported alongside the hit rates.
"""

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.gold import contains_span, load_heldout  # noqa: E402

RESULTS_DIR = REPO_ROOT / "eval" / "results"


def gdvae_rag():
    from retrieval.hybrid_search import hybrid_search
    from retrieval.rerank import rerank

    def retrieve(question: str) -> list[dict]:
        return [{"id": c["id"], "text": c["text"]} for c in rerank(question, hybrid_search(question), top_k=5)]

    return retrieve, None


def stair_zeroshot():
    from stair.toc import load_toc
    from stair.zeroshot import ZeroShotToCRetriever

    retriever = ZeroShotToCRetriever(load_toc())

    def retrieve(question: str) -> list[dict]:
        return [{"id": leaf.id, "text": leaf.text} for leaf in retriever.retrieve(question)]

    return retrieve, retriever.usage


def _local_toc_retriever(adapter_path):
    from stair.finetuned import FineTunedToCRetriever
    from stair.toc import load_toc

    retriever = FineTunedToCRetriever(load_toc(), adapter_path)

    def retrieve(question: str) -> list[dict]:
        return [{"id": leaf.id, "text": leaf.text} for leaf in retriever.retrieve(question)]

    return retrieve, None


def stair_base():
    # The untrained base model, to show how much the fine-tune adds (the paper's Mistral row).
    return _local_toc_retriever(None)


def stair_finetuned():
    return _local_toc_retriever(str(REPO_ROOT / "data" / "stair" / "adapters"))


SYSTEMS = {
    "gdvae-rag": gdvae_rag,
    "stair-zeroshot": stair_zeroshot,
    "stair-base": stair_base,
    "stair-finetuned": stair_finetuned,
}


def evaluate(system: str) -> None:
    retrieve, usage = SYSTEMS[system]()
    rows = []
    for q in load_heldout():
        results = retrieve(q["question"])
        hits = [contains_span(r["text"], q["gold_span"]) for r in results]
        first_hit = next((i + 1 for i, h in enumerate(hits) if h), None)
        rows.append({
            "id": q["id"],
            "category": q["category"],
            "first_hit_rank": first_hit,
            "retrieved": [r["id"] for r in results],
            "words_returned": sum(len(r["text"].split()) for r in results),
        })

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{system}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    n = len(rows)
    k = max(len(r["retrieved"]) for r in rows)
    at1 = sum(r["first_hit_rank"] == 1 for r in rows)
    atk = sum(r["first_hit_rank"] is not None for r in rows)
    print(f"{system}: Hit@1 {at1}/{n} = {at1 / n:.1%}   Hit@{k} {atk}/{n} = {atk / n:.1%}")
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        hit = sum(r["first_hit_rank"] is not None for r in sub)
        print(f"  {cat:6s} Hit@{k} {hit}/{len(sub)}")
    print(f"  words returned per question: median {statistics.median(r['words_returned'] for r in rows):.0f}")
    if usage:
        print(f"  tokens: {usage}")
    print(f"  per-question results -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SYSTEMS:
        sys.exit(f"usage: python eval/run_heldout.py {{{'|'.join(SYSTEMS)}}}")
    evaluate(sys.argv[1])
