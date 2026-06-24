"""
RAG Document Retrieval API
FastAPI app with three endpoints:
  POST /upload   — parse + index a document
  POST /query    — retrieve + generate answer
  GET  /health   — health check
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Literal
from dotenv import load_dotenv

load_dotenv()  # loads .env locally; no effect when platform injects env vars

from app.parser import extract_text
from app.rag import index_document, retrieve_chunks, doc_id_from_filename, list_documents
from app.llm import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Startup validation — fail fast if secrets are missing ─────────────────────

REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"  Local dev: add them to your .env file\n"
            f"  Render:    Dashboard → Environment → Add variable\n"
            f"  Railway:   Dashboard → Variables → Add variable"
        )
    logger.info("All required environment variables present. Starting server.")
    yield
    logger.info("Server shutting down.")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    lifespan=lifespan,
    title="RAG Document Retrieval API",
    description=(
        "Upload PDF / DOCX / TXT documents and query them using "
        "Retrieval-Augmented Generation with Groq (Llama 3)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "RAG Document Retrieval API"}


@app.get("/documents")
def get_documents():
    """List all indexed document IDs."""
    return {"documents": list_documents()}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document (PDF, DOCX, or TXT).
    The document is parsed, chunked, embedded, and stored in ChromaDB.
    """
    allowed = {".pdf", ".docx", ".txt"}
    filename = file.filename or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Upload PDF, DOCX, or TXT.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="File too large. Max size is 20 MB.")

    try:
        text = extract_text(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if len(text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Extracted text is too short. Check that the document has readable content.",
        )

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
    """
    Query an indexed document.

    Modes:
    - summary  → concise 3-5 sentence summary
    - outline  → structured bullet-point outline
    - full     → all key details in structured format
    - qa       → answer a specific question (requires 'question' field)
    """
    doc_id = doc_id_from_filename(body.filename)

    # For full/summary/outline mode, use a descriptive query to retrieve broad content
    retrieval_query = (
        body.question
        if body.mode == "qa"
        else f"Main topics, key points, and important details of the document"
    )

    try:
        chunks = retrieve_chunks(
            query=retrieval_query,
            doc_id=doc_id,
            top_k=8 if body.mode == "full" else 5,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found in document.")

    try:
        answer = generate_answer(
            chunks=chunks,
            mode=body.mode,
            question=body.question,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return QueryResponse(
        filename=body.filename,
        mode=body.mode,
        question=body.question,
        chunks_retrieved=len(chunks),
        answer=answer,
    )
