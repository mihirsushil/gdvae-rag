"""Shared BM25 tokenizer. Must stay identical between index build and query time."""

import re

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
