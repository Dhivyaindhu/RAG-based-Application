"""
RAG pipeline — chunk text, embed, store in ChromaDB, retrieve relevant chunks.

Strategy used: Parent-Child RAG
- Small chunks (200 words) embedded and searched for precision.
- Parent chunks (600 words) returned to LLM for richer context.

ChromaDB 0.6.x — uses PersistentClient(path=...)
"""

import os
import uuid
import hashlib
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Render mounts the persistent disk at this path (set via env var in render.yaml)
# Falls back to ./vectorstore for local development
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")

# ── Singletons ────────────────────────────────────────────────────────────────
_embed_model = None
_chroma_client = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    return _chroma_client


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_text_large(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


# ── Helpers ───────────────────────────────────────────────────────────────────

def doc_id_from_filename(filename: str) -> str:
    safe = hashlib.md5(filename.encode()).hexdigest()[:16]
    return f"doc_{safe}"


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_document(text: str, filename: str) -> Dict:
    doc_id = doc_id_from_filename(filename)
    client = get_chroma_client()
    model = get_embed_model()

    try:
        client.delete_collection(doc_id)
    except Exception:
        pass

    collection = client.create_collection(
        name=doc_id,
        metadata={"hnsw:space": "cosine"},
    )

    child_chunks = chunk_text(text)
    parent_chunks = chunk_text_large(text)

    child_words_seen = 0
    child_to_parent: List[int] = []
    total_words = max(len(text.split()), 1)

    for child in child_chunks:
        parent_idx = min(
            int(child_words_seen / total_words * len(parent_chunks)),
            len(parent_chunks) - 1,
        )
        child_to_parent.append(parent_idx)
        child_words_seen += len(child.split())

    embeddings = model.encode(child_chunks, show_progress_bar=False).tolist()

    collection.add(
        ids=[str(uuid.uuid4()) for _ in child_chunks],
        embeddings=embeddings,
        documents=child_chunks,
        metadatas=[
            {
                "child_index": i,
                "parent_text": parent_chunks[child_to_parent[i]],
                "filename": filename,
            }
            for i in range(len(child_chunks))
        ],
    )
    # PersistentClient auto-persists in chromadb 0.6.x — no manual persist() needed

    return {
        "doc_id": doc_id,
        "child_chunks": len(child_chunks),
        "parent_chunks": len(parent_chunks),
        "filename": filename,
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, doc_id: str, top_k: int = 6) -> List[str]:
    client = get_chroma_client()
    model = get_embed_model()

    try:
        collection = client.get_collection(doc_id)
    except Exception:
        raise ValueError("Document not found. Please upload it first.")

    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    seen, parent_chunks = set(), []
    for meta in results["metadatas"][0]:
        key = meta["parent_text"][:80]
        if key not in seen:
            seen.add(key)
            parent_chunks.append(meta["parent_text"])

    return parent_chunks


def list_documents() -> List[str]:
    return [c.name for c in get_chroma_client().list_collections()]
