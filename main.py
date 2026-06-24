"""
RAG Document Retrieval API
FastAPI app:
  GET  /          — HTML frontend
  POST /upload    — parse + index a document
  POST /query     — retrieve + generate answer
  GET  /health    — health check
  GET  /documents — list indexed documents
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

from doc_parser import extract_text
from rag import index_document, retrieve_chunks, doc_id_from_filename, list_documents
from llm import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ["GROQ_API_KEY"]
STATIC_DIR = Path(__file__).parent / "static"


# ── Startup validation ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"  Local dev: add them to your .env file\n"
            f"  Render:    Dashboard → Environment → Add variable"
        )
    logger.info("All required environment variables present. Starting server.")
    yield
    logger.info("Server shutting down.")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    lifespan=lifespan,
    title="RAG Document Retrieval API",
    description="Upload PDF / DOCX / TXT and query with Groq Llama 3.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML frontend)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    filename: str
    mode: Literal["summary", "outline", "full", "qa"]
    question: str = ""

    @field_validator("question")
    @classmethod
    def qa_requires_question(cls, v, info):
        if info.data.get("mode") == "qa" and not v.strip():
            raise ValueError("mode='qa' requires a non-empty question.")
        return v


class UploadResponse(BaseModel):
    message: str
    filename: str
    doc_id: str
    child_chunks: int
    parent_chunks: int


class QueryResponse(BaseModel):
    filename: str
    mode: str
    question: str
    chunks_retrieved: int
    answer: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Serve the HTML frontend."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "RAG API is running. Visit /docs for the API reference."}


@app.get("/health")
def health():
    return {"status": "ok", "service": "RAG Document Retrieval API"}


@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".txt"}
    filename = file.filename or "upload"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if ext not in allowed:
        raise HTTPException(400, detail=f"Unsupported file type '{ext}'. Use PDF, DOCX, or TXT.")

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, detail="File too large. Max 20 MB.")

    try:
        text = extract_text(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))

    if len(text.strip()) < 50:
        raise HTTPException(422, detail="Extracted text too short. Check the document has readable content.")

    result = index_document(text, filename)

    return UploadResponse(
        message="Document indexed successfully.",
        filename=filename,
        doc_id=result["doc_id"],
        child_chunks=result["child_chunks"],
        parent_chunks=result["parent_chunks"],
    )


@app.post("/query", response_model=QueryResponse)
async def query_document(body: QueryRequest):
    doc_id = doc_id_from_filename(body.filename)

    retrieval_query = (
        body.question if body.mode == "qa"
        else "Main topics, key points, and important details of the document"
    )

    try:
        chunks = retrieve_chunks(
            query=retrieval_query,
            doc_id=doc_id,
            top_k=8 if body.mode == "full" else 5,
        )
    except ValueError as e:
        raise HTTPException(404, detail=str(e))

    if not chunks:
        raise HTTPException(404, detail="No relevant content found.")

    try:
        answer = generate_answer(chunks=chunks, mode=body.mode, question=body.question)
    except EnvironmentError as e:
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"LLM error: {str(e)}")

    return QueryResponse(
        filename=body.filename,
        mode=body.mode,
        question=body.question,
        chunks_retrieved=len(chunks),
        answer=answer,
    )
