"""Build a corpus-wide table of contents: one leaf per PDF outline entry, markdown heading, or top-level Python def/class."""

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
TOC_PATH = REPO_ROOT / "data" / "stair" / "toc.json"

HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


@dataclass
class Leaf:
    id: str  # "L001" — short, stable within one build
    source: str
    path: list[str]  # breadcrumb from document down to this leaf
    text: str


_LATEX_JUNK = re.compile(r"bold0mu\s*mumu\s*|\d{4}/\d\d/\d\d ver: [\d.]+ subfig package")
REFERENCES_RE = re.compile(r"\n\s*(References|Bibliography)\s*\n")


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", _LATEX_JUNK.sub("", title)).strip()


def _flatten_outline(reader: PdfReader, outline, depth=0, parents=()):
    """Yield (title, start_page, parents, is_leaf) in reading order."""
    items = list(outline)
    for i, item in enumerate(items):
        if isinstance(item, list):
            continue
        has_children = i + 1 < len(items) and isinstance(items[i + 1], list)
        title = _clean_title(item.title)
        yield title, reader.get_destination_page_number(item), parents, not has_children
        if has_children:
            yield from _flatten_outline(reader, items[i + 1], depth + 1, parents + (title,))


def _cut_at_title(page_text: str, title: str) -> int:
    """Offset of the heading inside its start page, so a leaf doesn't swallow the previous section's tail."""
    words = title.split()
    if not words:
        return 0
    pattern = r"\s*".join(re.escape(w) for w in words[-min(len(words), 4):])
    m = re.search(pattern, page_text)
    return m.start() if m else 0


def pdf_leaves(path: Path, source: str) -> list[tuple[list[str], str]]:
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    entries = list(_flatten_outline(reader, reader.outline))
    out = []
    first_page = entries[0][1]
    front = "\n".join(pages[:first_page] + [pages[first_page][: _cut_at_title(pages[first_page], entries[0][0])]])
    if front.strip():
        out.append(([path.name, "(front matter: title, abstract, introduction)"], front.strip()))
    for i, (title, start, parents, is_leaf) in enumerate(entries):
        nxt = entries[i + 1] if i + 1 < len(entries) else None
        end_page = nxt[1] if nxt else len(pages) - 1
        start_off = _cut_at_title(pages[start], title)
        if nxt and end_page == start:
            end_off = _cut_at_title(pages[start], nxt[0]) or len(pages[start])
            text = pages[start][start_off:end_off]
        else:
            parts = [pages[start][start_off:]] + pages[start + 1 : end_page]
            if nxt:
                parts.append(pages[end_page][: _cut_at_title(pages[end_page], nxt[0])])
            else:
                parts.append(pages[end_page])
            text = "\n".join(parts)
        if not nxt:
            text = REFERENCES_RE.split(text)[0]
        # Parent headings with children get no text of their own beyond the intro; keep only leaves.
        if is_leaf:
            out.append(([path.name, *parents, title], text.strip()))
    return out


def markdown_leaves(text: str, source: str) -> list[tuple[list[str], str]]:
    headers = list(HEADER_RE.finditer(text))
    out = []
    if not headers:
        return [([source], text.strip())] if text.strip() else []
    if text[: headers[0].start()].strip():
        out.append(([source, "(preamble)"], text[: headers[0].start()].strip()))
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[m.end() : end].strip()
        if body:
            out.append(([source, m.group(2).strip()], body))
    return out


def python_leaves(path: Path, source: str) -> list[tuple[list[str], str]]:
    text = path.read_text(errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not nodes:
        return [([source], text.strip())] if text.strip() else []
    out = []
    for n in nodes:
        seg = ast.get_source_segment(text, n)
        if seg:
            kind = "class" if isinstance(n, ast.ClassDef) else "def"
            out.append(([source, f"{kind} {n.name}"], seg))
    return out


def build_toc() -> list[Leaf]:
    raw: list[tuple[str, list[str], str]] = []
    repo_dir = RAW_DIR / "gd-vae-repo"
    for md in sorted(repo_dir.rglob("*.md")):
        src = f"gd-vae-repo/{md.relative_to(repo_dir)}"
        raw += [(src, p, t) for p, t in markdown_leaves(md.read_text(errors="ignore"), src)]
    for py in sorted(repo_dir.rglob("*.py")):
        src = f"gd-vae-repo/{py.relative_to(repo_dir)}"
        raw += [(src, p, t) for p, t in python_leaves(py, src)]
    for pdf in sorted((RAW_DIR / "papers").glob("*.pdf")):
        src = f"papers/{pdf.name}"
        raw += [(src, p, t) for p, t in pdf_leaves(pdf, src)]
    return [Leaf(f"L{i:03d}", src, path, text) for i, (src, path, text) in enumerate(raw, start=1)]


def render_toc(leaves: list[Leaf]) -> str:
    """The table of contents exactly as a model sees it: one line per leaf."""
    return "\n".join(f"{leaf.id} {' > '.join(leaf.path)}" for leaf in leaves)


def load_toc() -> list[Leaf]:
    return [Leaf(**d) for d in json.loads(TOC_PATH.read_text())]


def main() -> None:
    leaves = build_toc()
    TOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOC_PATH.write_text(json.dumps([asdict(l) for l in leaves], indent=1))
    words = [len(l.text.split()) for l in leaves]
    print(f"wrote {len(leaves)} leaves to {TOC_PATH}; words/leaf median {sorted(words)[len(words)//2]}, max {max(words)}")


if __name__ == "__main__":
    main()
