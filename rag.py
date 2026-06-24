"""
RAG pipeline — lightweight version for Render free tier (512MB RAM)

Embedding strategy: Groq does NOT provide embeddings, so we use
the `chromadb` default embedding function which calls sentence-transformers
ONLY when needed, OR we use a hash-based TF-IDF style retrieval.

To stay within 512MB we replace sentence-transformers with
chromadb's built-in lightweight embedding (all-MiniLM-L6-v2 via ONNX runtime
which chromadb bundles — much lighter than full PyTorch).
"""

import os
import uuid
import hashlib
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")

_chroma_client = None
_embed_fn = None


def get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        # Uses chromadb's built-in ONNX embedding — no PyTorch needed
        # Model: all-MiniLM-L6-v2 via ONNX (~30MB vs ~400MB for PyTorch)
        _embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return _embed_fn


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


def doc_id_from_filename(filename: str) -> str:
    safe = hashlib.md5(filename.encode()).hexdigest()[:16]
    return f"doc_{safe}"


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_document(text: str, filename: str) -> Dict:
    doc_id = doc_id_from_filename(filename)
    client = get_chroma_client()

    try:
        client.delete_collection(doc_id)
    except Exception:
        pass

    # Pass embedding function at collection creation
    collection = client.create_collection(
        name=doc_id,
        embedding_function=get_embed_fn(),
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

    # chromadb handles embedding internally — just pass documents
    collection.add(
        ids=[str(uuid.uuid4()) for _ in child_chunks],
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

    return {
        "doc_id": doc_id,
        "child_chunks": len(child_chunks),
        "parent_chunks": len(parent_chunks),
        "filename": filename,
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, doc_id: str, top_k: int = 6) -> List[str]:
    client = get_chroma_client()

    try:
        collection = client.get_collection(
            doc_id,
            embedding_function=get_embed_fn(),
        )
    except Exception:
        raise ValueError("Document not found. Please upload it first.")

    results = collection.query(
        query_texts=[query],
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
