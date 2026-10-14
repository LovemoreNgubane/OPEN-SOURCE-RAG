"""Query the vector store with a natural-language question.

This is the open-source equivalent of the tutorial's similarity_search.py.
"""
import logging

from app.database.vector_store import VectorStore
from app.services.embeddings import embed_text

logging.basicConfig(level=logging.INFO)


def main() -> None:
    store = VectorStore()

    # A couple of test questions - replace with your own once data is loaded
    questions = [
        "What is the quadratic formula?",
        "Explain Newton's second law of motion.",
        "What is the weather in Tokyo?",   # irrelevant - should score low
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        query_vector = embed_text(question)
        results = store.search(query_vector, limit=3)

        if not results:
            print("No results found. Did you run the ingestion step first?")
            continue

        for i, r in enumerate(results, start=1):
            print(f"\n[{i}] similarity={r['similarity']:.4f}")
            print(f"    id: {r['id']}")
            print(f"    contents: {r['contents'][:200]}...")
            print(f"    metadata: {r['metadata']}")


if __name__ == "__main__":
    main()