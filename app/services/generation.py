"""Llama 3.2 generation via Ollama (Docker).

Open-source, local, no API key.
"""
import logging

import ollama

from app.config.settings import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Grade 12 tutor for South African students.

Your job: read the Context below, then answer the Question using that Context.

RULES:
1. If the Context contains the answer, USE IT and give the answer directly.
2. If the Context does NOT contain the answer, reply exactly: "I don't have enough information to answer that."
3. Do not invent facts. Do not add information from outside the Context.
4. Keep answers under 150 words. Be clear and encouraging.

Examples:
- Context mentions "F = ma" and Question asks "What is Newton's second law?" -> Answer: "Newton's second law states that force equals mass times acceleration, F = ma."
- Context is about math formulas and Question asks "What is the weather in Tokyo?" -> Answer: "I don't have enough information to answer that."
"""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Generate an answer to `question` using retrieved `context_chunks`."""
    if not context_chunks:
        return "I don't have enough information to answer that."

    # Build the context block with numbered references
    context_text = "\n\n".join(
        f"[{i+1}] {chunk['contents']}"
        for i, chunk in enumerate(context_chunks)
    )

    user_prompt = f"""Context:
{context_text}

Question: {question}"""

    logger.info(f"Calling {settings.llm_model} via Ollama...")

    try:
        client = ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"Generation failed: {e}"