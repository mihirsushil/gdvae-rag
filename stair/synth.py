"""Synthetic training questions for STAIR: Claude reads each leaf section and writes questions it answers.

Writes data/stair/synth.jsonl, one row per question: {"leaf_id", "question"}. Questions that closely echo a
held-out eval question are dropped, so the fine-tune never trains on the test set.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.gold import load_heldout  # noqa: E402
from stair.toc import Leaf, load_toc  # noqa: E402

MODEL = "claude-haiku-5-5"
QUESTIONS_PER_LEAF = 8
MAX_LEAF_WORDS = 4000
OVERLAP_LIMIT = 0.6  # Jaccard over content words; above this a question counts as a near-copy of a held-out one
SYNTH_PATH = REPO_ROOT / "data" / "stair" / "synth.jsonl"

INSTRUCTIONS = f"""You write training questions for a retriever over a research corpus about GD-VAE \
(geometric dynamic variational autoencoders): its code, docs and related papers.

You are given one section of the corpus and where it sits in the table of contents. Write \
{QUESTIONS_PER_LEAF} questions that this section answers. Mix the kinds: specific facts, how or why \
something works, what a function or parameter does, and comparisons. Write them the way a researcher or \
developer would actually ask, without quoting long phrases from the section and without naming the section \
title. Each question must be answerable from this section."""

SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}

STOPWORDS = set("a an the of in on for to and or is are what which how why does do by with from as at be this that it its".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def generate(client: anthropic.Anthropic, leaf: Leaf) -> tuple[list[str], anthropic.types.Usage]:
    text = " ".join(leaf.text.split()[:MAX_LEAF_WORDS])
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=INSTRUCTIONS,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"Location: {' > '.join(leaf.path)}\n\nSection text:\n{text}"}],
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"{leaf.id}: unexpected stop_reason {response.stop_reason!r} (request {response._request_id})")
    questions = json.loads(next(b.text for b in response.content if b.type == "text"))["questions"]
    return [q.strip() for q in questions if q.strip()][:QUESTIONS_PER_LEAF], response.usage


def main() -> None:
    leaves = load_toc()
    heldout = [_content_words(q["question"]) for q in load_heldout()]
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda leaf: generate(client, leaf), leaves))

    rows, dropped = [], 0
    usage = {"input": 0, "output": 0}
    for leaf, (questions, u) in zip(leaves, results):
        usage["input"] += u.input_tokens
        usage["output"] += u.output_tokens
        for q in questions:
            if max(_jaccard(_content_words(q), h) for h in heldout) > OVERLAP_LIMIT:
                dropped += 1
                continue
            rows.append({"leaf_id": leaf.id, "question": q})

    SYNTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNTH_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} questions for {len(leaves)} leaves to {SYNTH_PATH.relative_to(REPO_ROOT)}")
    print(f"  dropped {dropped} near-copies of held-out questions (Jaccard > {OVERLAP_LIMIT})")
    print(f"  tokens: {usage}")


if __name__ == "__main__":
    main()
