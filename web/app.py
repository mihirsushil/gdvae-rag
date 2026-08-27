"""Browser demo for the gdvae-rag retrieval pipeline — same search() logic as the MCP server, over HTTP.

Adds one thing the MCP server doesn't do: a plain-language paraphrase of the retrieved
passages (web/summarize.py), rate-limited per visitor since it spends real API credits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank
from web.summarize import summarize

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

EXAMPLE_QUERIES = [
    "How does GD-VAE encode geometric structure into the latent space?",
    "What loss terms does the VAE training objective combine?",
    "How is the encoder architecture defined in the source code?",
]

SUMMARIES_PER_VISITOR = 2
_summary_counts: dict[str, int] = {}  # in-memory, resets on restart — a budget guard, not a security control


async def index(request: Request) -> HTMLResponse:
    query = request.query_params.get("q", "").strip()
    results = None
    summary = None
    rate_limited = False
    if query:
        candidates = hybrid_search(query)
        results = rerank(query, candidates, top_k=5)

        client_ip = request.client.host if request.client else "unknown"
        used = _summary_counts.get(client_ip, 0)
        if used < SUMMARIES_PER_VISITOR:
            summary = summarize(query, results)
            if summary is not None:
                _summary_counts[client_ip] = used + 1
        else:
            rate_limited = True

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": query,
            "results": results,
            "summary": summary,
            "rate_limited": rate_limited,
            "examples": EXAMPLE_QUERIES,
        },
    )


app = Starlette(routes=[Route("/", index)])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
