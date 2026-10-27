"""Gold-span matching shared by every retriever's eval, so all systems are scored by the same rule."""

import json
import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout.jsonl"


def normalize(text: str) -> str:
    """Lowercase alphanumerics only: pypdf drops spaces and splits hyphenated words unpredictably."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text).lower())


def contains_span(text: str, span: str) -> bool:
    """Whole span, or either half when a chunk boundary cuts it (halves are long enough to stay distinctive)."""
    t, s = normalize(text), normalize(span)
    if s in t:
        return True
    half = len(s) // 2
    return half >= 30 and (s[:half] in t or s[half:] in t)


def load_heldout() -> list[dict]:
    return [json.loads(line) for line in HELDOUT_PATH.read_text().splitlines() if line.strip()]


def gold_leaf_ids(span: str, leaves) -> set[str]:
    return {leaf.id for leaf in leaves if normalize(span) in normalize(leaf.text)}
