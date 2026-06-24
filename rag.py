"""
RAG pipeline — chunk text, embed, store in ChromaDB, retrieve relevant chunks.

Strategy used: Parent-Child RAG
- Small chunks (200 tokens) are embedded and searched for precision.
- Parent chunks (600 tokens) are returned to the LLM for richer context.

ChromaDB version: 0.5.23 (pinned for Render free tier compatibility)
"""

import uuid
import hashlib
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Embedding model (runs locally, no API key needed) ───────────────────────
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 80MB, fast, good quality
_embed_model = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


# ── ChromaDB client — 0.5.x API (persists to disk via duckdb+parquet) ───────
# chromadb 0.5.x uses chromadb.Client(Settings(...)) for persistence.
# chromadb 1.x changed to PersistentClient() — do NOT upgrade without
# also rewriting this section.
_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./vectorstore",
                anonymized_telemetry=False,
            )
        )
    return _chroma_client


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """
    Split text into word-based chunks with overlap.
    chunk_size: words per small (child) chunk — used for embedding/search.
    overlap: words shared between consecutive chunks to preserve context.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_text_large(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
    """Larger (parent) chunks — sent to LLM for answer generation."""
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


# ── Indexing ──────────────────────────────────────────────────────────────────

def doc_id_from_filename(filename: str) -> str:
    """Stable collection name derived from filename (ChromaDB safe)."""
    safe = hashlib.md5(filename.encode()).hexdigest()[:16]
    return f"doc_{safe}"


def index_document(text: str, filename: str) -> Dict:
    """
    Chunk, embed, and store a document in ChromaDB.
    Returns metadata: doc_id, child_chunks, parent_chunks.

    Parent-Child strategy:
    - child_chunks: small, precise → embedded and stored in Chroma
    - parent_chunks: large, contextual → stored as metadata alongside child
    """
    doc_id = doc_id_from_filename(filename)
    client = get_chroma_client()
    model = get_embed_model()

    # Delete existing collection for this doc (re-upload = re-index)
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

    # Map each child chunk to its nearest parent chunk index
    child_to_parent: List[int] = []
    child_words_seen = 0
    for i, child in enumerate(child_chunks):
        child_word_count = len(child.split())
        parent_idx = min(
            int(child_words_seen / max(len(text.split()), 1) * len(parent_chunks)),
            len(parent_chunks) - 1,
        )
        child_to_parent.append(parent_idx)
        child_words_seen += child_word_count

    # Embed all child chunks at once (batched)
    embeddings = model.encode(child_chunks, show_progress_bar=False).tolist()

    ids = [str(uuid.uuid4()) for _ in child_chunks]
    metadatas = [
        {
            "child_index": i,
            "parent_text": parent_chunks[child_to_parent[i]],
            "filename": filename,
        }
        for i in range(len(child_chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=child_chunks,
        metadatas=metadatas,
    )

    # Persist to disk (required in chromadb 0.5.x)
    client.persist()

    return {
        "doc_id": doc_id,
        "child_chunks": len(child_chunks),
        "parent_chunks": len(parent_chunks),
        "filename": filename,
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, doc_id: str, top_k: int = 6) -> List[str]:
    """
    Embed the query, retrieve top-k child chunks by cosine similarity,
    return the corresponding PARENT chunks (for richer LLM context).
    Deduplicates parents so the same section isn't sent twice.
    """
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

    # Deduplicate parent chunks preserving retrieval order
    seen_parents: set = set()
    parent_chunks: List[str] = []
    for meta in results["metadatas"][0]:
        parent_text = meta["parent_text"]
        key = parent_text[:80]  # fingerprint
        if key not in seen_parents:
            seen_parents.add(key)
            parent_chunks.append(parent_text)

    return parent_chunks


def list_documents() -> List[str]:
    """Return names of all indexed collections (document IDs)."""
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]
