# RAG-based-Application
first task of retreival Based application
# RAG Document Retrieval API

A production-ready Retrieval-Augmented Generation (RAG) app built with:

- **FastAPI** — REST API backend
- **ChromaDB** — local vector store (persisted to disk)
- **sentence-transformers** — local embedding model (no API key needed)
- **Groq (Llama 3 70B)** — LLM reader + writer (free tier)
- **Parent-Child RAG** — small chunks for search precision, large chunks for LLM context

Supports **PDF, DOCX, and TXT** documents with four output modes:
`summary` · `outline` · `full` · `qa`

---

## Project structure

```
rag-doc-app/
├── app/
│   ├── __init__.py
│   ├── main.py        ← FastAPI endpoints
│   ├── parser.py      ← PDF / DOCX / TXT text extraction
│   ├── rag.py         ← chunking, embedding, ChromaDB, retrieval
│   └── llm.py         ← Groq client + prompt templates
├── vectorstore/       ← ChromaDB data (auto-created, gitignored)
├── uploads/           ← optional local upload cache (gitignored)
├── requirements.txt
├── render.yaml        ← Render deployment config
├── Procfile           ← Railway deployment config
├── .env.example
└── .gitignore
```

---

## Local setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd rag-doc-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free Groq API key

1. Go to https://console.groq.com
2. Sign up (free, no credit card)
3. Create an API key

### 3. Set your environment variable

```bash
cp .env.example .env
# Edit .env and paste your key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

API is now live at: http://localhost:8000
Interactive docs: http://localhost:8000/docs

---

## API usage

### Upload a document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "message": "Document indexed successfully.",
  "filename": "your_document.pdf",
  "doc_id": "doc_a1b2c3d4e5f6g7h8",
  "child_chunks": 42,
  "parent_chunks": 14
}
```

---

### Query — Summary

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"filename": "your_document.pdf", "mode": "summary"}'
```

---

### Query — Outline

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"filename": "your_document.pdf", "mode": "outline"}'
```

---

### Query — Full detail

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"filename": "your_document.pdf", "mode": "full"}'
```

---

### Query — Q&A (ask a specific question)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "your_document.pdf",
    "mode": "qa",
    "question": "What are the main conclusions of this document?"
  }'
```

Response (all modes):
```json
{
  "filename": "your_document.pdf",
  "mode": "qa",
  "question": "What are the main conclusions?",
  "chunks_retrieved": 5,
  "answer": "The document concludes that..."
}
```

---

## RAG pipeline explained

```
Upload flow:
  File bytes → parser.py (extract text)
             → rag.py (chunk into 200-word child chunks)
             → sentence-transformers (embed each chunk)
             → ChromaDB (store vectors + parent chunk as metadata)

Query flow:
  User query → sentence-transformers (embed query)
             → ChromaDB (cosine similarity → top-8 child chunks)
             → Fetch parent chunks (600-word context windows)
             → llm.py (mode-based prompt + Groq Llama 3 70B)
             → Answer streamed back
```

**Why Parent-Child RAG?**
Small chunks (200 words) are precise for search — they match the query well.
But 200 words is too little context for the LLM to write a good answer.
So after retrieving the right small chunks, we send their larger parent chunks
(600 words) to the LLM. Best of both worlds: precision in retrieval, richness in generation.

---

## Deployment

### Option A — Render (recommended, free)

1. Push code to GitHub
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Add environment variable: `GROQ_API_KEY` = your key
6. Deploy — Render provides a persistent disk for ChromaDB

### Option B — Railway

1. Push code to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add environment variable: `GROQ_API_KEY` = your key
4. Railway uses the `Procfile` automatically
5. Note: Railway free tier has limited persistent storage — use an external
   vector DB like Qdrant Cloud (free tier) for production

### Option C — Vercel (not recommended for this app)

Vercel is serverless — ChromaDB needs a persistent filesystem, which
serverless functions don't support between invocations. Use Render instead.

---

## Upgrading the stack

| Component | Current | Upgrade to |
|-----------|---------|------------|
| Embedding | all-MiniLM-L6-v2 (local) | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB (local disk) | Qdrant Cloud / Pinecone |
| LLM | Groq Llama 3 70B | GPT-4o / Claude |
| RAG type | Parent-Child | Add HyDE or re-ranking for harder queries |

---

## Adding re-ranking (optional upgrade)

Install: `pip install sentence-transformers`

In `rag.py`, after retrieval add:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[str], top_n: int = 3) -> list[str]:
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]
```

Call `rerank(query, chunks)` before passing to `generate_answer()`.
This adds ~200ms but meaningfully improves answer quality.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Free at console.groq.com |
