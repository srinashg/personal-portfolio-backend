# personal-portfolio-backend

Minimal Python RAG backend for portfolio/resume Q&A.

## Features

- FastAPI service with:
  - `GET /health`
  - `POST /ingest` (build/rebuild index)
  - `POST /ask` (RAG answer)
- Local FAISS vector index
- OpenAI embeddings + chat model
- Markdown data sources under `data/`

## Quick start

1. Create and activate a virtual env
2. Install deps
3. Configure `.env`
4. Run ingest
5. Start server

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
cp .env.example .env
python -m app.ingest
uvicorn app.main:app --reload --port 8000
```

## API

### Health
```bash
curl http://localhost:8000/health
```

### Ask
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are Srinesh’s strongest frontend skills?"}'
```

### Re-index
```bash
curl -X POST http://localhost:8000/ingest
```

## Notes

- Keep `data/*.md` up to date with your portfolio content.
- `index/` is generated and gitignored.
- If answer quality is weak, tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.
