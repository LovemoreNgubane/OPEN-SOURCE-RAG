"""Full RAG pipeline: retrieve context, then generate with Llama.

Usage:
    uv run python -m app.services.rag_pipeline
"""
import logging

from app.database.vector_store import VectorStore
from app.services.embeddings import embed_text
from app.services.generation import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ask(question: str, top_k: int = 3) -> dict:
    """Run the full RAG pipeline for a question.

    1. Embed the question
    2. Retrieve top-K similar documents from pgvector
    3. Send question + retrieved context to Llama via Ollama
    4. Return answer plus source citations
    """
    logger.info(f"Question: {question}")

    # 1. Embed the question
    query_vector = embed_text(question)

    # 2. Retrieve similar documents
    store = VectorStore()
    chunks = store.search(query_vector, limit=top_k)
    logger.info(f"Retrieved {len(chunks)} chunks")

    # 3. Generate answer with Llama
    answer = generate_answer(question, chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "id": c["id"],
                "category": c["metadata"].get("category"),
                "similarity": c["similarity"],
                "preview": c["contents"][:100],
            }
            for c in chunks
        ],
    }


def main() -> None:
    """CLI test with three questions."""
    questions = [
        "What is the quadratic formula?",
        "Explain Newton's second law.",
        "What is the weather in Tokyo?",   # irrelevant - should be refused
    ]

    for q in questions:
        print("\n" + "=" * 70)
        result = ask(q)
        print(f"QUESTION: {result['question']}")
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSources used:")
        for s in result["sources"]:
            print(f"  - [{s['category']}] similarity={s['similarity']:.4f}")


if __name__ == "__main__":
    main()