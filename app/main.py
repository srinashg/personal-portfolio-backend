from fastapi import FastAPI, HTTPException

from app.ingest import run_ingest
from app.models import AskRequest, AskResponse, HealthResponse
from app.rag import answer_question


app = FastAPI(title="Portfolio RAG Backend", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/ingest")
def ingest():
    try:
        return run_ingest()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        answer, sources = answer_question(req.question)
        return AskResponse(answer=answer, sources=sources)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Index not found. Call /ingest first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
