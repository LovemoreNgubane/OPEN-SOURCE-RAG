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

MIN_SIMILARITY = 0.15   # drop clearly-irrelevant chunks
MAX_CONTEXT = 3         # how many chunks to send to Llama


def ask(question: str, top_k: int = 10) -> dict:
    """Run the full RAG pipeline for a question.

    1. Embed the question
    2. Retrieve top-K similar documents from pgvector
    3. Filter out noise below MIN_SIMILARITY
    4. Isolate to the top source (prevents FAQ/PDF cross-contamination)
    5. Send question + context to Llama via Ollama
    """
    logger.info(f"Question: {question}")

    # 1. Embed the question
    query_vector = embed_text(question)

    # 2. Retrieve more candidates than we'll use
    store = VectorStore()
    candidates = store.search(query_vector, limit=top_k)
    logger.info(f"Retrieved {len(candidates)} candidates")

    # 3. Filter out noise
    filtered = [c for c in candidates if c["similarity"] >= MIN_SIMILARITY]
    logger.info(f"After filter (>= {MIN_SIMILARITY}): {len(filtered)} chunks")

    # 4. Isolate to the top source - prevents FAQ chunks from confusing
    #    the LLM when the question is really about a PDF (and vice-versa).
    if filtered:
        top_source = (
            filtered[0]["metadata"].get("source")
            or filtered[0]["metadata"].get("category")
        )
        filtered = [
            c for c in filtered
            if (
                c["metadata"].get("source") or c["metadata"].get("category")
            ) == top_source
        ]
        logger.info(f"After source isolation ({top_source}): {len(filtered)} chunks")

    # 5. Keep only the best few
    chunks = filtered[:MAX_CONTEXT]

    # 6. Generate answer with Llama
    answer = generate_answer(question, chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "id": c["id"],
                "source": c["metadata"].get("source") or c["metadata"].get("category"),
                "similarity": c["similarity"],
                "preview": c["contents"][:120],
            }
            for c in chunks
        ],
    }


def main() -> None:
    """CLI test with questions about the AI Academy PDF."""
    questions = [
        "What is the AI Academy South Africa?",
        "Who can apply to the Pitchathon?",
        "What technologies must the solution use?",
    ]

    for q in questions:
        print("\n" + "=" * 70)
        result = ask(q)
        print(f"QUESTION: {result['question']}")
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSources used:")
        for s in result["sources"]:
            print(f"  - [{s['source']}] similarity={s['similarity']:.4f}")
            print(f"      {s['preview']}...")


if __name__ == "__main__":
    main()