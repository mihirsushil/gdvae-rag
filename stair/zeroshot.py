"""Zero-shot STAIR baseline: Claude reads the corpus table of contents and picks the 3 most likely leaves.

No fine-tuning — this mirrors the paper's Claude Haiku row, not STAIR itself. Structured output with an
enum of leaf ids plays the role of STAIR's constrained decoding: the model cannot name a section that
doesn't exist.
"""

import json

import anthropic

from stair.toc import Leaf, render_toc

MODEL = "claude-haiku-5-5"
TOP_K = 3

INSTRUCTIONS = (
    "You are a retriever over a research corpus: the GD-VAE code repository, its docs, the GD-VAE paper, "
    "Kingma & Welling's VAE paper, and a geometric deep learning survey. Below is the corpus table of "
    "contents, one leaf section per line as '<id> <path>'. For the user's question, pick the leaf sections "
    "whose text most likely contains the answer, best first. Answer only with the ids."
)


def _schema(leaf_ids: list[str]) -> dict:
    # Fixed slots instead of an array: array length constraints aren't supported in structured outputs.
    slot = {"type": "string", "enum": leaf_ids}
    names = ["first", "second", "third"][:TOP_K]
    return {
        "type": "object",
        "properties": {n: slot for n in names},
        "required": names,
        "additionalProperties": False,
    }


class ZeroShotToCRetriever:
    def __init__(self, leaves: list[Leaf], client: anthropic.Anthropic | None = None):
        self.leaves = {leaf.id: leaf for leaf in leaves}
        self.client = client or anthropic.Anthropic()
        # The ToC is identical on every request, so it sits in a cached system block ahead of the question.
        self.system = [
            {"type": "text", "text": INSTRUCTIONS},
            {"type": "text", "text": render_toc(leaves), "cache_control": {"type": "ephemeral"}},
        ]
        self.output_config = {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": _schema(sorted(self.leaves))},
        }
        self.usage = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}

    def retrieve(self, question: str) -> list[Leaf]:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=self.system,
            output_config=self.output_config,
            messages=[{"role": "user", "content": question}],
        )
        u = response.usage
        self.usage["input"] += u.input_tokens
        self.usage["cache_read"] += u.cache_read_input_tokens or 0
        self.usage["cache_write"] += u.cache_creation_input_tokens or 0
        self.usage["output"] += u.output_tokens
        if response.stop_reason != "end_turn":
            raise RuntimeError(f"unexpected stop_reason {response.stop_reason!r} (request {response._request_id})")
        picks = json.loads(next(b.text for b in response.content if b.type == "text"))
        ids = list(dict.fromkeys(picks[k] for k in ["first", "second", "third"][:TOP_K]))
        return [self.leaves[i] for i in ids]
