"""GD-VAE research RAG — MCP server backed by a custom hybrid search + rerank pipeline."""

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader

from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"

mcp = FastMCP("gdvae-rag")


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """Search the GD-VAE corpus (repo code/docs + background papers) for passages relevant to a question.

    Args:
        query: Natural language question or search query.
        top_k: Number of ranked, cited passages to return (default 5).
    """
    candidates = hybrid_search(query)
    top = rerank(query, candidates, top_k=top_k)
    if not top:
        return "No relevant passages found."

    parts = []
    for i, chunk in enumerate(top, start=1):
        location = chunk["source"]
        if chunk.get("section"):
            location += f" § {chunk['section']}"
        if chunk.get("page"):
            location += f" (page {chunk['page']})"
        parts.append(f"[{i}] {location}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def get_document(source: str) -> str:
    """Return the full text of one source document from the corpus.

    Args:
        source: Source identifier as returned by search() or list_sources(),
            e.g. "gd-vae-repo/src/vae.py" or "papers/gdvae_paper.pdf".
    """
    if source.startswith("papers/"):
        pdf_path = RAW_DIR / source
        if not pdf_path.exists():
            return f"Unknown source: {source}"
        reader = PdfReader(str(pdf_path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    file_path = RAW_DIR / source
    if not file_path.exists():
        return f"Unknown source: {source}"
    return file_path.read_text(errors="ignore")


@mcp.tool()
def list_sources() -> str:
    """List every source document in the corpus, with its type and chunk count."""
    with CHUNKS_PATH.open() as f:
        chunks = [json.loads(line) for line in f]

    by_source: dict[str, dict] = {}
    for chunk in chunks:
        info = by_source.setdefault(chunk["source"], {"doc_type": chunk["doc_type"], "count": 0})
        info["count"] += 1

    lines = [f"{source} ({info['doc_type']}, {info['count']} chunks)" for source, info in sorted(by_source.items())]
    return "\n".join(lines)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(os.environ.get("PORT", 8000))
        mcp.settings.stateless_http = True

        external_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
        if external_host:
            mcp.settings.transport_security.allowed_hosts += [external_host, f"{external_host}:*"]
            mcp.settings.transport_security.allowed_origins += [f"https://{external_host}"]

    mcp.run(transport=transport)
