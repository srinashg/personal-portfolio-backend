from pathlib import Path
from typing import List, Dict

DATA_DIR = Path("data")


def load_markdown_documents() -> List[Dict[str, str]]:
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": str(path), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks
