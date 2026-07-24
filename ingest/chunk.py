"""Chunking: header-aware for markdown/PDF text, AST-based for Python source."""

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"

WINDOW_WORDS = 400
OVERLAP_WORDS = 60

HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    source: str  # relative path or paper filename
    doc_type: str  # "markdown" | "pdf" | "python"
    section: str  # heading / function name / "" for unstructured windows
    page: int | None  # page number for PDFs, else None
    text: str


def _sliding_windows(text: str, words: int = WINDOW_WORDS, overlap: int = OVERLAP_WORDS):
    tokens = text.split()
    if not tokens:
        return
    step = words - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + words]
        if not window:
            break
        yield " ".join(window)
        if start + words >= len(tokens):
            break


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    headers = list(HEADER_RE.finditer(text))

    if not headers:
        for i, window in enumerate(_sliding_windows(text)):
            chunks.append(Chunk(f"{source}::w{i}", source, "markdown", "", None, window))
        return chunks

    if headers[0].start() > 0:
        preamble = text[: headers[0].start()].strip()
        if preamble:
            chunks.append(Chunk(f"{source}::preamble", source, "markdown", "", None, preamble))

    for idx, match in enumerate(headers):
        section_title = match.group(2).strip()
        start = match.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        if len(section_text.split()) <= WINDOW_WORDS:
            chunks.append(
                Chunk(f"{source}::{section_title}", source, "markdown", section_title, None, f"{section_title}\n{section_text}")
            )
        else:
            for i, window in enumerate(_sliding_windows(section_text)):
                chunks.append(
                    Chunk(f"{source}::{section_title}::w{i}", source, "markdown", section_title, None, f"{section_title}\n{window}")
                )
    return chunks


def chunk_pdf(path: Path, source: str) -> list[Chunk]:
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for i, window in enumerate(_sliding_windows(text)):
            chunks.append(Chunk(f"{source}::p{page_num}::w{i}", source, "pdf", "", page_num, window))
    return chunks


def chunk_python(path: Path, source: str) -> list[Chunk]:
    text = path.read_text(errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    top_level_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not top_level_nodes:
        return [Chunk(f"{source}::w{i}", source, "python", "", None, w) for i, w in enumerate(_sliding_windows(text))]

    chunks: list[Chunk] = []
    for node in top_level_nodes:
        segment = ast.get_source_segment(text, node)
        if not segment:
            continue
        chunks.append(Chunk(f"{source}::{node.name}", source, "python", node.name, None, segment))
    return chunks


def collect_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []

    repo_dir = RAW_DIR / "gd-vae-repo"
    for md_path in sorted(repo_dir.rglob("*.md")):
        rel = md_path.relative_to(repo_dir)
        chunks.extend(chunk_markdown(md_path.read_text(errors="ignore"), source=f"gd-vae-repo/{rel}"))

    for py_path in sorted(repo_dir.rglob("*.py")):
        rel = py_path.relative_to(repo_dir)
        chunks.extend(chunk_python(py_path, source=f"gd-vae-repo/{rel}"))

    papers_dir = RAW_DIR / "papers"
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        chunks.extend(chunk_pdf(pdf_path, source=f"papers/{pdf_path.name}"))

    return chunks


def main() -> None:
    chunks = collect_chunks()
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    print(f"wrote {len(chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
