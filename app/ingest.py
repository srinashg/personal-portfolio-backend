from app.config import settings
from app.loaders import load_markdown_documents, chunk_text
from app.rag import embed_texts, save_index


def run_ingest() -> dict:
    docs = load_markdown_documents()
    if not docs:
        raise RuntimeError("No markdown files found in data/")

    metadata = []
    all_chunks = []

    for d in docs:
        chunks = chunk_text(
            d["text"],
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        for c in chunks:
            all_chunks.append(c)
            metadata.append({"source": d["source"], "chunk": c})

    vectors = embed_texts(all_chunks)
    save_index(vectors, metadata)

    return {"documents": len(docs), "chunks": len(all_chunks)}


if __name__ == "__main__":
    result = run_ingest()
    print(f"Ingest complete: {result}")
