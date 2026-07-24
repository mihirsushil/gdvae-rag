"""Download the corpus: the gd-vae repo and the three background papers."""

import subprocess
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REPO_URL = "https://github.com/gd-vae/gd-vae.git"

PAPERS = {
    "gdvae_paper.pdf": "https://arxiv.org/pdf/2206.05183",
    "vae_kingma_welling.pdf": "https://arxiv.org/pdf/1312.6114",
    "geometric_deep_learning_survey.pdf": "https://arxiv.org/pdf/2104.13478",
}


def fetch_repo() -> None:
    repo_dir = RAW_DIR / "gd-vae-repo"
    if repo_dir.exists():
        print(f"repo already present at {repo_dir}, skipping clone")
        return
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)


def fetch_papers() -> None:
    papers_dir = RAW_DIR / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in PAPERS.items():
        dest = papers_dir / filename
        if dest.exists():
            print(f"{filename} already present, skipping")
            continue
        print(f"downloading {url} -> {dest}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_repo()
    fetch_papers()
    print("done")
