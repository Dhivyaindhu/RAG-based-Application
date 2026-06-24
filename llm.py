"""
LLM module — Groq (Llama 3) as the reader + writer.
Mode-based prompt templates control what the LLM produces from retrieved chunks.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Groq client ───────────────────────────────────────────────────────────────
_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Model config ──────────────────────────────────────────────────────────────
GROQ_MODEL = "llama3-70b-8192"  # Best quality on free tier
MAX_TOKENS = 2048


# ── Prompt templates per output mode ─────────────────────────────────────────

SYSTEM_PROMPTS = {
    "summary": (
        "You are a document summariser. "
        "Read the provided document excerpts and write a clear, concise summary "
        "in 3 to 5 sentences. Capture only the most important ideas. "
        "Do not add information not present in the excerpts."
    ),
    "outline": (
        "You are a document analyst. "
        "Read the provided document excerpts and produce a structured outline. "
        "Use bullet points with clear headings. "
        "Organise from high-level topics down to supporting points. "
        "Do not add information not present in the excerpts."
    ),
    "full": (
        "You are a document reader. "
        "Read the provided document excerpts and reproduce all key details "
        "in a well-structured, readable format. "
        "Use headings and paragraphs. Be thorough — do not omit important information. "
        "Do not add information not present in the excerpts."
    ),
    "qa": (
        "You are a precise question-answering assistant. "
        "Answer the user's question using only the provided document excerpts. "
        "Be direct and cite relevant details from the text. "
        "If the answer is not in the excerpts, say: "
        "'I could not find this in the document.'"
    ),
}

USER_PROMPT_TEMPLATES = {
    "summary": (
        "Here are excerpts from the document:\n\n"
        "{context}\n\n"
        "Write a summary of this document."
    ),
    "outline": (
        "Here are excerpts from the document:\n\n"
        "{context}\n\n"
        "Produce a structured outline of this document."
    ),
    "full": (
        "Here are excerpts from the document:\n\n"
        "{context}\n\n"
        "Reproduce all the key details from these excerpts in a well-structured format."
    ),
    "qa": (
        "Here are excerpts from the document:\n\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer the question based on the excerpts above."
    ),
}


# ── Main generate function ────────────────────────────────────────────────────

def generate_answer(
    chunks: list[str],
    mode: str,
    question: str = "",
) -> str:
    """
    Send retrieved chunks to Groq LLM and generate an answer.

    Args:
        chunks: Retrieved document chunks (from RAG pipeline)
        mode: One of 'summary', 'outline', 'full', 'qa'
        question: Required when mode == 'qa'

    Returns:
        LLM-generated text response
    """
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid mode '{mode}'. Choose: summary, outline, full, qa")

    if mode == "qa" and not question.strip():
        raise ValueError("mode='qa' requires a non-empty question.")

    context = "\n\n---\n\n".join(chunks)

    user_prompt = USER_PROMPT_TEMPLATES[mode].format(
        context=context,
        question=question,
    )

    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # Low temperature → factual, consistent output
    )

    return response.choices[0].message.content.strip()
