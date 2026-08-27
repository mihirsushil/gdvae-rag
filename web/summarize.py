"""Plain-language layer on top of the retrieval pipeline: paraphrases the already-cited
passages into a short, grounded answer. The retrieval step above this is what makes the
paraphrase trustworthy — the model is never allowed to add anything not in the passages."""

import os

from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300

SYSTEM_PROMPT = (
    "You explain dense academic passages in plain, conversational English for a "
    "non-expert reader. Use ONLY the numbered passages given below — never add "
    "outside knowledge or guess at details not present in them. Cite claims inline "
    "like [1] or [2]. If the passages don't actually answer the question, say so "
    "plainly instead of guessing. Keep the whole answer to 3-5 sentences."
)

_client: Anthropic | None = None


def _get_client() -> Anthropic | None:
    global _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    if _client is None:
        _client = Anthropic(api_key=api_key)
    return _client


def summarize(query: str, chunks: list[dict]) -> str | None:
    """Paraphrase retrieved chunks into a short, cited, plain-language answer.

    Returns None if no API key is configured or the call fails — callers should
    fall back to showing the raw cited passages only.
    """
    client = _get_client()
    if client is None or not chunks:
        return None

    numbered = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Question: {query}\n\nPassages:\n\n{numbered}"}],
        )
        return response.content[0].text.strip()
    except Exception:
        return None
