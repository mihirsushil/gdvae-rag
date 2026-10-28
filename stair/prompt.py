"""The STAIR input/output format shared by dataset building and inference: instruction + ToC + question -> leaf path."""

from stair.toc import Leaf

INSTRUCTION = (
    "Below is the table of contents of a research corpus, as an indented tree of files and sections. "
    "Answer with the full path of the one leaf section that best answers the question, joined with ' > '."
)


def user_message(toc: str, question: str) -> str:
    return f"{INSTRUCTION}\n\n{toc}\n\nQuestion: {question}"


def target(leaf: Leaf) -> str:
    return " > ".join(leaf.path)
