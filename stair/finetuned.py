"""Fine-tuned STAIR retriever: a LoRA-tuned small model scores every leaf path given the ToC and the question.

Scoring all 163 valid paths and keeping the best 3 is constrained decoding taken to its limit: the model can
only ever return a section that exists. The prompt is encoded once per question; each candidate path is run
on top of that cached prompt and trimmed off again.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from stair.prompt import target, user_message
from stair.toc import Leaf, render_toc_tree

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
TOP_K = 3


class FineTunedToCRetriever:
    def __init__(self, leaves: list[Leaf], adapter_path: str | None):
        self.model, self.tokenizer = load(BASE_MODEL, adapter_path=adapter_path)
        self.toc = render_toc_tree(leaves)
        eos = self.tokenizer.eos_token_id
        self.candidates = [(leaf, self.tokenizer.encode(target(leaf), add_special_tokens=False) + [eos]) for leaf in leaves]

    def _prompt_tokens(self, question: str) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message(self.toc, question)}], tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def scores(self, question: str) -> list[tuple[float, Leaf]]:
        cache = make_prompt_cache(self.model)
        first = nn.log_softmax(self.model(mx.array(self._prompt_tokens(question))[None], cache=cache)[0, -1])
        out = []
        for leaf, toks in self.candidates:
            logprobs = nn.log_softmax(self.model(mx.array(toks[:-1])[None], cache=cache)[0], axis=-1)
            total = first[toks[0]] + logprobs[mx.arange(len(toks) - 1), mx.array(toks[1:])].sum()
            out.append((total.item(), leaf))
            trim_prompt_cache(cache, len(toks) - 1)
        return sorted(out, key=lambda s: s[0], reverse=True)

    def retrieve(self, question: str) -> list[Leaf]:
        return [leaf for _, leaf in self.scores(question)[:TOP_K]]
