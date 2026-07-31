"""Regression tests for BM25 tokenization and retrieval. Run: .venv/bin/python3 tests/test_retrieval.py"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.tokenize import tokenize  # noqa: E402
from retrieval.hybrid_search import sparse_search, _load_chunks  # noqa: E402


def test_identifier_next_to_punctuation_is_its_own_token():
    tokens = tokenize("def map_clifford_torus(input,params):")
    assert "map_clifford_torus" in tokens, tokens
    assert "map_clifford_torus(input,params):" not in tokens, tokens


def test_underscore_identifiers_survive_tokenization():
    assert tokenize("gd_vae.geo_map.map_clifford_torus") == [
        "gd_vae",
        "geo_map",
        "map_clifford_torus",
    ]


def test_exact_identifier_query_retrieves_its_definition():
    chunks = _load_chunks()
    top = sparse_search("map_clifford_torus", top_n=5)
    sources = [chunks[i]["source"] for i in top]
    assert any("geo_map.py" in s for s in sources), sources


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} passed")
