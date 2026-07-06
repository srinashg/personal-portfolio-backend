import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI

from app.config import settings


INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "vectors.faiss"
META_FILE = INDEX_DIR / "metadata.json"


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: List[str]) -> np.ndarray:
    client = _client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    vectors = np.array([d.embedding for d in response.data], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def save_index(vectors: np.ndarray, metadata: List[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index_and_metadata():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Index not found. Run ingest first.")
    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, metadata


def retrieve(question: str, top_k: int) -> List[dict]:
    index, metadata = load_index_and_metadata()
    q_vec = embed_texts([question])
    scores, indices = index.search(q_vec, top_k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        if i < 0:
            continue
        item = metadata[i].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def answer_question(question: str) -> Tuple[str, List[str]]:
    hits = retrieve(question, settings.top_k)
    if not hits:
        return "I don’t have enough information in the portfolio data to answer that.", []

    context_blocks = []
    source_set = set()
    for i, h in enumerate(hits, start=1):
        source_set.add(h["source"])
        context_blocks.append(f"[Source {i}: {h['source']}]\n{h['chunk']}")

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a portfolio assistant for Srinesh Ganesh. "
        "Answer ONLY from the provided context. "
        "If context is insufficient, clearly say you do not have enough info. "
        "Be concise and factual."
    )

    user_prompt = f"""Question:
{question}

Context:
{context}
"""

    client = _client()
    resp = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = resp.choices[0].message.content or "I couldn’t generate an answer."
    return answer, sorted(source_set)
