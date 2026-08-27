FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install the CPU-only torch build first — the default PyPI wheel bundles CUDA
# and is ~10x larger, which blows past small-instance memory/disk budgets for
# a corpus this size that never touches a GPU. Since requirements.txt pins no
# torch version, the later install is satisfied by this one and skips it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY server.py README.md ./
COPY retrieval/ retrieval/
COPY web/ web/
COPY data/index.faiss data/chunks.jsonl data/bm25.pkl data/

# Overridden per-service (web demo vs. remote MCP) via each service's start command.
# Shell form so $PORT (injected by the host platform) actually expands.
CMD uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000}
