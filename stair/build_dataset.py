"""Turn data/stair/synth.jsonl into chat-format train/valid files for mlx_lm. Each leaf's last question goes to valid."""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stair.prompt import target, user_message  # noqa: E402
from stair.synth import SYNTH_PATH  # noqa: E402
from stair.toc import load_toc, render_toc_tree  # noqa: E402

SFT_DIR = REPO_ROOT / "data" / "stair" / "sft"


def main() -> None:
    leaves = {leaf.id: leaf for leaf in load_toc()}
    toc = render_toc_tree(list(leaves.values()))
    by_leaf = defaultdict(list)
    for line in SYNTH_PATH.read_text().splitlines():
        row = json.loads(line)
        by_leaf[row["leaf_id"]].append(row["question"])

    splits = {"train": [], "valid": []}
    for leaf_id, questions in by_leaf.items():
        for i, q in enumerate(questions):
            example = {"messages": [
                {"role": "user", "content": user_message(toc, q)},
                {"role": "assistant", "content": target(leaves[leaf_id])},
            ]}
            splits["valid" if i == len(questions) - 1 else "train"].append(example)

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (SFT_DIR / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"train {len(splits['train'])}, valid {len(splits['valid'])} -> {SFT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
