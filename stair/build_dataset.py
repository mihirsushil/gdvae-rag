"""Turn synthetic questions into chat-format train/valid files for mlx_lm.

Usage: python stair/build_dataset.py [--more]

Each leaf's last question in data/stair/synth.jsonl goes to valid, the rest to train. With --more, the second
batch (data/stair/synth_more.jsonl) is added to train only, so valid stays the same 163 questions across runs
and the runs can be compared on it.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stair.prompt import target, user_message  # noqa: E402
from stair.synth import SYNTH_MORE_PATH, SYNTH_PATH  # noqa: E402
from stair.toc import load_toc, render_toc_tree  # noqa: E402

SFT_DIR = REPO_ROOT / "data" / "stair" / "sft"
SFT_MORE_DIR = REPO_ROOT / "data" / "stair" / "sft_more"


def _by_leaf(path: Path) -> dict[str, list[str]]:
    by_leaf = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        by_leaf[row["leaf_id"]].append(row["question"])
    return by_leaf


def main() -> None:
    more = sys.argv[1:] == ["--more"]
    out_dir = SFT_MORE_DIR if more else SFT_DIR
    leaves = {leaf.id: leaf for leaf in load_toc()}
    toc = render_toc_tree(list(leaves.values()))

    def example(leaf_id: str, question: str) -> dict:
        return {"messages": [
            {"role": "user", "content": user_message(toc, question)},
            {"role": "assistant", "content": target(leaves[leaf_id])},
        ]}

    splits = {"train": [], "valid": []}
    for leaf_id, questions in _by_leaf(SYNTH_PATH).items():
        for i, q in enumerate(questions):
            splits["valid" if i == len(questions) - 1 else "train"].append(example(leaf_id, q))
    if more:
        for leaf_id, questions in _by_leaf(SYNTH_MORE_PATH).items():
            splits["train"] += [example(leaf_id, q) for q in questions]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"train {len(splits['train'])}, valid {len(splits['valid'])} -> {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
