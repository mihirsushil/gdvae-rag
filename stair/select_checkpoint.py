"""Pick the best saved LoRA checkpoint by leaf accuracy on the synthetic valid set, and copy it to <adapters>/best.

Usage: python stair/select_checkpoint.py ADAPTER_DIR SFT_DIR

Per-token validation loss hid run 1's real problem (right file, wrong section), so checkpoints are compared on
the retrieval task itself: does the correct leaf rank first, or in the top 3? Only synthetic valid questions are
used, never the held-out eval set, so choosing a checkpoint cannot leak the test set into the result.
"""

import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stair.finetuned import TOP_K, FineTunedToCRetriever, base_model  # noqa: E402
from stair.prompt import target  # noqa: E402
from stair.toc import load_toc  # noqa: E402

EVERY_NTH = 4  # 41 of the 163 valid questions


def _valid_questions(sft_dir: Path) -> list[tuple[str, str]]:
    """(question, gold leaf path) for every EVERY_NTH valid example."""
    rows = []
    for line in (sft_dir / "valid.jsonl").read_text().splitlines()[::EVERY_NTH]:
        user, assistant = json.loads(line)["messages"]
        rows.append((user["content"].rsplit("\n\nQuestion: ", 1)[1], assistant["content"]))
    return rows


def _checkpoints(adapter_dir: Path) -> list[tuple[str, Path]]:
    found = [(p.name.split("_")[0].lstrip("0"), p) for p in sorted(adapter_dir.glob("*_adapters.safetensors"))]
    return found + [("final", adapter_dir / "adapters.safetensors")]


def main() -> None:
    adapter_dir, sft_dir = Path(sys.argv[1]), Path(sys.argv[2])
    leaves = load_toc()
    questions = _valid_questions(sft_dir)
    model = base_model(str(adapter_dir))
    staging = adapter_dir / "_staging"

    results = []
    for step, weights in _checkpoints(adapter_dir):
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        shutil.copy(adapter_dir / "adapter_config.json", staging)
        (staging / "adapters.safetensors").symlink_to(weights.resolve())

        start = time.time()
        retriever = FineTunedToCRetriever(leaves, model, str(staging))
        ranks = []
        for question, gold in questions:
            ranked = [target(leaf) for _, leaf in retriever.scores(question)]
            ranks.append(ranked.index(gold) + 1)
        at1 = sum(r == 1 for r in ranks)
        atk = sum(r <= TOP_K for r in ranks)
        results.append({"step": step, "weights": weights.name, "hit_at_1": at1, f"hit_at_{TOP_K}": atk, "n": len(ranks)})
        print(f"checkpoint {step:>5}: Hit@1 {at1}/{len(ranks)}  Hit@{TOP_K} {atk}/{len(ranks)}  ({time.time() - start:.0f}s)", flush=True)
        del retriever

    shutil.rmtree(staging, ignore_errors=True)
    # Ties go to the later checkpoint: it has seen more data for the same valid score.
    best = max(reversed(results), key=lambda r: (r[f"hit_at_{TOP_K}"], r["hit_at_1"]))
    best_dir = adapter_dir / "best"
    shutil.rmtree(best_dir, ignore_errors=True)
    best_dir.mkdir()
    shutil.copy(adapter_dir / "adapter_config.json", best_dir)
    shutil.copy(adapter_dir / best["weights"], best_dir / "adapters.safetensors")
    (adapter_dir / "checkpoint_scores.json").write_text(json.dumps({"best": best["step"], "checkpoints": results}, indent=2))
    print(f"best: checkpoint {best['step']} -> {best_dir}")


if __name__ == "__main__":
    main()
