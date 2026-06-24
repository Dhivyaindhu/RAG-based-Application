"""
Document parser — extracts clean text from PDF, DOCX, and TXT files.
"""

import io
from pathlib import Path


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from uploaded file bytes.
    Supports: .pdf, .docx, .txt
    Returns the full extracted text as a single string.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(file_bytes)
    elif ext == ".docx":
        return _extract_docx(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")


def _extract_pdf(file_bytes: bytes) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())
    doc.close()

    if not pages:
        raise ValueError("PDF appears to be empty or image-only (no extractable text).")

    return "\n\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        raise ValueError("DOCX file appears to be empty.")

    return "\n\n".join(paragraphs)
